#!/usr/bin/env python3
"""Strict verifier for a packed-NVFP4 PLE candidate checkpoint.

Conforms to ``docs/rvn-ple-storage-schema.md`` v1. Every rule of that contract
is exactly one named check, evaluated in the fixed order below; the JSON
report lists every check with ``pass``/``fail`` plus a reason, and the exit
code is 0 if and only if every check passes.

    manifest_schema        manifest keys, types and fixed values (contract §2)
    global_scale_bits      raw uint32 decodes to a finite positive float (§1/§2)
    partitioning           numeric-ordered, complete, non-overlapping cover (§2)
    parts_integrity        parts cover the same rows, rows > 0, payload digests
                           match the stored bytes (§2/§4)
    source_table_sha256    recomputed from the source checkpoint (§4)
    non_ple_identity       every retained non-PLE tensor keeps dtype, shape and
                           payload digest; file boundaries may move (§4)
    index_refs             model.safetensors.index.json resolves (§2)
    partial_candidate      no manifest-without-part and no part-without-manifest;
                           there is no BF16 fallback semantics (§2)
    dequant_sample         bounded row re-decode: the §1 reference path and the
                           loader path must agree bit-for-bit (§1)

Quantised-versus-source-BF16 difference is expected and is NOT checked here.
Only stdlib, torch (dequant loader path) and structurally-parsed safetensors
headers are used; no tensor is ever materialised whole for hashing, and only
sampled rows are read for the dequant check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path, PurePosixPath

TOOL_NAME = "rvn_ple.verify"
TOOL_VERSION = "1.0.0"

MANIFEST_NAME = "ple_storage.json"
INDEX_NAME = "model.safetensors.index.json"

FORMAT_VERSION = 1
REQUIRED_LOADER_FEATURE = "ple-packed-nvfp4-v1"
WEIGHT_DTYPE = "e2m1-packed-u8-low-first"
SCALE_DTYPE = "float8_e4m3fn"
SCALE_LAYOUT = "row-major"
RECONSTRUCTION = "bf16_direct"
GROUP_SIZE = 16
SOURCE_DTYPE = "bfloat16"
# safetensors stores the manifest's logical dtype names under short codes.
_STORE_DTYPE = {"bfloat16": "BF16", "float32": "F32", "float16": "F16"}
# safetensors store codes for the packed tensors of contract §1.
WEIGHT_STORE_DTYPE = "U8"
SCALE_STORE_DTYPE = "F8_E4M3"

CHUNK = 1 << 20
DEFAULT_DEQUANT_SAMPLE = 8

# Contract §1: E2M1 nibble magnitudes, index = nibble & 0x07.
_E2M1_MAGNITUDE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_PACKED_TENSOR_RE = re.compile(r"^rvn_ple\.packed\.[ws]\d+$")


class VerifyError(Exception):
    """A contract-mandated input is missing, unreadable or malformed."""


def _reject_duplicate_keys(pairs):
    """JSON hook rejecting duplicate keys (a manifest must not double-declare)."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require(condition, message):
    if not condition:
        raise VerifyError(message)


def _read_shard_header(path: Path):
    """Parse one safetensors header; return ``{name: entry}``, no tensor data."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise VerifyError(f"unreadable shard (truncated header): {path}")
            (header_len,) = struct.unpack("<Q", head)
            raw = handle.read(header_len)
            if len(raw) != header_len:
                raise VerifyError(f"unreadable shard (truncated header JSON): {path}")
        meta = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except VerifyError:
        raise
    except (OSError, ValueError) as exc:
        raise VerifyError(f"unreadable shard: {path}: {exc}") from exc
    # safetensors data_offsets are relative to the end of the header.
    payload_base = 8 + header_len
    _require(isinstance(meta, dict), f"unreadable shard (bad header): {path}")

    tensors = {}
    for name, entry in meta.items():
        if name == "__metadata__":
            continue
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("dtype"), str)
            and isinstance(entry.get("shape"), list)
            and all(isinstance(dim, int) and dim >= 0 for dim in entry["shape"]),
            f"unreadable shard (bad tensor entry {name!r}): {path}",
        )
        offsets = entry.get("data_offsets")
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(off, int) and off >= 0 for off in offsets)
            and offsets[0] <= offsets[1],
            f"unreadable shard (bad data_offsets for {name!r}): {path}",
        )
        tensors[name] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [offsets[0] + payload_base, offsets[1] + payload_base],
        }
    return tensors


def _shards(root: Path):
    """``{shard relative name: header}`` for every shard below ``root``."""
    _require(root.is_dir(), f"missing checkpoint directory: {root}")
    found = {}
    for path in sorted(root.rglob("*.safetensors")):
        if not path.is_file():
            continue
        found[path.relative_to(root).as_posix()] = _read_shard_header(path)
    return found


def _resolve_shard(shards, rel_name, *, where):
    """Resolve a manifest-declared shard name, tolerating directory prefixes."""
    _require(isinstance(rel_name, str) and rel_name != "", f"{where}: empty shard name")
    if rel_name in shards:
        return rel_name
    base = PurePosixPath(rel_name).name
    matches = [name for name in shards if PurePosixPath(name).name == base]
    _require(
        len(matches) == 1,
        f"{where}: shard {rel_name!r} is {'missing' if not matches else 'ambiguous'}",
    )
    return matches[0]


def _hash_payload(digest, path: Path, start, end):
    """Feed ``[start, end)`` of a shard's stored payload into ``digest``."""
    _require(end >= start, f"bad payload range {start}..{end} in {path}")
    remaining = end - start
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise VerifyError(
                    f"truncated payload: {path} ends before byte {end}"
                )
            digest.update(block)
            remaining -= len(block)


def _sha256_payload(path: Path, start, end) -> str:
    digest = hashlib.sha256()
    _hash_payload(digest, path, start, end)
    return digest.hexdigest()


def _read_payload(path: Path, start, end) -> bytes:
    """Read one bounded payload range (a single sampled row's bytes)."""
    _require(end >= start, f"bad payload range {start}..{end} in {path}")
    wanted = end - start
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(wanted)
    _require(
        len(data) == wanted, f"truncated payload: {path} ends before byte {end}"
    )
    return data


def _leading_rows(shape):
    """Row count of a logical table tensor: product of the leading dims."""
    _require(len(shape) >= 2, f"tensor shape {shape} is not a row-major table")
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows


def _f32(value):
    """Round a Python float to IEEE-754 binary32 (overflow -> inf, per IEEE)."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _bf16_rne_bits(value):
    """Signed int32 bits of the single BF16 round-to-nearest-even of ``value``."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<i", struct.pack("<I", bits))[0]


def _e4m3_to_f32(byte):
    """FP8 E4M3 byte to float; ``None`` for the NaN encoding (contract §1)."""
    sign, exponent, mantissa = byte >> 7, (byte >> 3) & 0xF, byte & 7
    if exponent == 0xF and mantissa == 7:
        return None
    if exponent == 0:
        value = (mantissa / 8.0) * 2.0**-6
    else:
        value = 2.0 ** (exponent - 7) * (1.0 + mantissa / 8.0)
    return -value if sign else value


def _validate_source(source):
    _require(isinstance(source, dict), "source must be an object")
    for key in ("repo", "revision"):
        # Provenance may legitimately be unknown to the converter, so only the
        # type is fixed here; a missing value is a string, not an absent key.
        _require(
            isinstance(source.get(key), str), f"source.{key} must be a string"
        )
    _require(
        _is_hex64(source.get("source_table_sha256")),
        "source.source_table_sha256 must be a lowercase sha256 hex digest",
    )
    _require(
        source.get("source_dtype") == SOURCE_DTYPE,
        f"source.source_dtype must be {SOURCE_DTYPE!r}",
    )
    _require(
        _is_num(source.get("amax")) and math.isfinite(source["amax"]),
        "source.amax must be a finite number",
    )


def _validate_table(table):
    _require(isinstance(table, dict), "table must be an object")
    _require(
        _is_int(table.get("logical_rows")) and table["logical_rows"] >= 1,
        "table.logical_rows must be an integer >= 1",
    )
    _require(
        _is_int(table.get("cols"))
        and table["cols"] >= GROUP_SIZE
        and table["cols"] % GROUP_SIZE == 0,
        f"table.cols must be an integer multiple of group_size {GROUP_SIZE}",
    )
    partitioning = table.get("partitioning")
    _require(
        isinstance(partitioning, list) and partitioning,
        "table.partitioning must be a non-empty list",
    )
    for index, entry in enumerate(partitioning):
        _require(isinstance(entry, dict), f"partitioning[{index}] must be an object")
        for key in ("part", "row_offset", "rows"):
            _require(
                _is_int(entry.get(key)),
                f"partitioning[{index}].{key} must be an integer",
            )
        for key in ("source_shard", "source_tensor"):
            _require(
                isinstance(entry.get(key), str) and entry[key] != "",
                f"partitioning[{index}].{key} must be a non-empty string",
            )


def _validate_encoding(encoding):
    _require(isinstance(encoding, dict), "encoding must be an object")
    fixed = {
        "weight_dtype": WEIGHT_DTYPE,
        "scale_dtype": SCALE_DTYPE,
        "scale_layout": SCALE_LAYOUT,
        "reconstruction": RECONSTRUCTION,
    }
    for key, want in fixed.items():
        _require(encoding.get(key) == want, f"encoding.{key} must be {want!r}")
    _require(
        _is_int(encoding.get("group_size")) and encoding["group_size"] == GROUP_SIZE,
        f"encoding.group_size must be {GROUP_SIZE}",
    )
    _require(
        _is_int(encoding.get("global_scale_bits"))
        and 0 <= encoding["global_scale_bits"] < 2**32,
        "encoding.global_scale_bits must be a uint32 integer",
    )


def _validate_parts(parts):
    _require(isinstance(parts, list) and parts, "parts must be a non-empty list")
    for index, part in enumerate(parts):
        _require(isinstance(part, dict), f"parts[{index}] must be an object")
        for key in (
            "file",
            "weight_tensor",
            "scale_tensor",
            "first_source_tensor",
            "last_source_tensor",
        ):
            _require(
                isinstance(part.get(key), str) and part[key] != "",
                f"parts[{index}].{key} must be a non-empty string",
            )
        for key in ("row_offset", "rows"):
            _require(
                _is_int(part.get(key)), f"parts[{index}].{key} must be an integer"
            )
        for key in ("sha256_weights", "sha256_scales"):
            _require(
                _is_hex64(part.get(key)),
                f"parts[{index}].{key} must be a lowercase sha256 hex digest",
            )


def _validate_manifest(meta):
    _require(isinstance(meta, dict), f"{MANIFEST_NAME} is not a JSON object")
    _require(
        _is_int(meta.get("format_version"))
        and meta["format_version"] == FORMAT_VERSION,
        f"format_version must be the integer {FORMAT_VERSION}",
    )
    _require(
        isinstance(meta.get("encoder_version"), str) and meta["encoder_version"] != "",
        "encoder_version must be a non-empty string",
    )
    _require(
        meta.get("required_loader_feature") == REQUIRED_LOADER_FEATURE,
        f"required_loader_feature must be {REQUIRED_LOADER_FEATURE!r}",
    )
    _validate_source(meta.get("source"))
    _validate_table(meta.get("table"))
    _validate_encoding(meta.get("encoding"))
    _validate_parts(meta.get("parts"))
    rewrites = meta.get("retained_rewrites")
    _require(
        isinstance(rewrites, dict)
        and all(
            isinstance(shard, str)
            and shard != ""
            and isinstance(names, list)
            and all(isinstance(name, str) and name != "" for name in names)
            for shard, names in rewrites.items()
        ),
        "retained_rewrites must map shard names to tensor-name lists",
    )


def _decode_global_scale(meta):
    bits = meta["encoding"]["global_scale_bits"]
    _require(
        _is_int(bits) and 0 <= bits < 2**32,
        "encoding.global_scale_bits must be a uint32 integer",
    )
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class _Ctx:
    """Lazy shared state for the checks; bad input always raises VerifyError."""

    def __init__(self, src_dir, dst_dir, dequant_sample):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.dequant_sample = dequant_sample
        self._manifest = None
        self._src_shards = None
        self._dst_shards = None

    def manifest(self):
        if self._manifest is None:
            path = self.dst_dir / MANIFEST_NAME
            _require(path.is_file(), f"missing {MANIFEST_NAME} in {self.dst_dir}")
            try:
                meta = json.loads(
                    path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
                )
            except (OSError, ValueError) as exc:
                raise VerifyError(f"unreadable {MANIFEST_NAME}: {exc}") from exc
            _validate_manifest(meta)
            self._manifest = meta
        return self._manifest

    def raw_manifest(self):
        """Unvalidated manifest, ``None`` when absent, ``"unreadable"`` on error."""
        path = self.dst_dir / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            return json.loads(
                path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
            )
        except (OSError, ValueError):
            return "unreadable"

    def src_shards(self):
        if self._src_shards is None:
            self._src_shards = _shards(self.src_dir)
        return self._src_shards

    def dst_shards(self):
        if self._dst_shards is None:
            self._dst_shards = _shards(self.dst_dir)
        return self._dst_shards

    def shard_path(self, root, rel_name):
        return root.joinpath(*PurePosixPath(rel_name).parts)

    def part_path(self, rel_name):
        pure = PurePosixPath(rel_name)
        _require(
            bool(pure.parts) and not pure.is_absolute() and ".." not in pure.parts,
            f"manifest part file escapes the candidate root: {rel_name!r}",
        )
        return self.shard_path(self.dst_dir, pure.as_posix())


def _check_manifest_schema(ctx):
    ctx.manifest()
    return True, "manifest keys, types and fixed values conform to contract §2"


def _check_global_scale_bits(ctx):
    global_scale = _decode_global_scale(ctx.manifest())
    if not math.isfinite(global_scale):
        return False, f"global_scale_bits decodes to non-finite {global_scale!r}"
    if global_scale <= 0.0:
        return False, f"global_scale_bits decodes to non-positive {global_scale!r}"
    return True, f"global_scale_bits decodes to {global_scale!r}"


def _check_partitioning(ctx):
    table = ctx.manifest()["table"]
    entries = table["partitioning"]
    part_ids = [entry["part"] for entry in entries]
    if len(set(part_ids)) != len(part_ids):
        return False, f"partitioning repeats part ids: {part_ids}"
    if part_ids != sorted(part_ids):
        return False, (
            "partitioning is not numeric-ordered (contract §2 forbids "
            f"lexicographic order): part sequence {part_ids}"
        )
    cursor = 0
    for entry in entries:
        if entry["rows"] <= 0:
            return False, f"partition part {entry['part']} has rows <= 0"
        if entry["row_offset"] != cursor:
            return False, (
                f"partitioning gap/overlap at part {entry['part']}: expected "
                f"row_offset {cursor}, found {entry['row_offset']}"
            )
        cursor += entry["rows"]
    if cursor != table["logical_rows"]:
        return False, (
            f"partitioning covers rows 0..{cursor - 1} but logical_rows is "
            f"{table['logical_rows']}"
        )
    return True, (
        f"{len(entries)} partitions cover rows 0..{cursor - 1} contiguously in "
        "numeric order"
    )


def _check_parts_integrity(ctx):
    meta = ctx.manifest()
    cols = meta["table"]["cols"]
    total = meta["table"]["logical_rows"]
    wanted = {
        "weight_tensor": (WEIGHT_STORE_DTYPE, (0, cols // 2)),
        "scale_tensor": (SCALE_STORE_DTYPE, (0, cols // GROUP_SIZE)),
    }
    spans = []
    for part in meta["parts"]:
        path = ctx.part_path(part["file"])
        _require(path.is_file(), f"part file missing: {part['file']}")
        header = _read_shard_header(path)
        for key, sha_key in (
            ("weight_tensor", "sha256_weights"),
            ("scale_tensor", "sha256_scales"),
        ):
            name = part[key]
            _require(
                name in header, f"part {part['file']} lacks tensor {name!r}"
            )
            entry = header[name]
            want_dtype, (_, want_cols) = wanted[key]
            want_shape = (part["rows"], want_cols)
            if entry["dtype"] != want_dtype or entry["shape"] != list(want_shape):
                return False, (
                    f"part {part['file']} tensor {name!r} is "
                    f"{entry['dtype']} {entry['shape']}, contract §1 requires "
                    f"{want_dtype} {list(want_shape)}"
                )
            start, end = entry["data_offsets"]
            digest = _sha256_payload(path, start, end)
            if digest != part[sha_key]:
                return False, (
                    f"sha256 {sha_key} mismatch for {part['file']}:{name}: "
                    f"manifest {part[sha_key][:16]}..., stored payload "
                    f"{digest[:16]}..."
                )
        spans.append((part["row_offset"], part["rows"]))
    spans.sort()
    cursor = 0
    for offset, rows in spans:
        if rows <= 0:
            return False, f"part at row_offset {offset} has rows <= 0"
        if offset != cursor:
            return False, (
                f"parts gap/overlap at row {cursor}: next part starts at {offset}"
            )
        cursor += rows
    if cursor != total:
        return False, (
            f"parts cover {cursor} rows but partitioning covers {total}"
        )
    return True, (
        f"{len(spans)} parts cover rows 0..{total - 1} with matching stored "
        "payload digests"
    )


def _source_table_digest(ctx, meta):
    """§4 digest over the source payload slice each partition covers.

    Entries splitting one source tensor must tile that tensor's rows, so the
    ascending-part concat covers the whole logical table exactly once.
    """
    table = meta["table"]
    shards = ctx.src_shards()
    wanted_dtype = _STORE_DTYPE[meta["source"]["source_dtype"]]
    entries = sorted(table["partitioning"], key=lambda entry: entry["part"])

    groups = {}
    for entry in entries:
        key = (entry["source_shard"], entry["source_tensor"])
        groups.setdefault(key, []).append(entry)

    resolved = {}
    for (shard_name, tensor_name), group in groups.items():
        where = f"partitioning part {min(entry['part'] for entry in group)}"
        rel = _resolve_shard(shards, shard_name, where=where)
        header = shards[rel]
        _require(
            tensor_name in header,
            f"{where}: source tensor {tensor_name!r} missing from shard {rel}",
        )
        declared = header[tensor_name]
        _require(
            declared["dtype"] == wanted_dtype,
            f"{where}: source tensor {tensor_name!r} is {declared['dtype']}, "
            f"source_dtype says {wanted_dtype}",
        )
        shape = declared["shape"]
        tensor_rows = _leading_rows(shape)
        _require(
            shape[-1] == table["cols"],
            f"{where}: source tensor {tensor_name!r} has {shape[-1]} columns, "
            f"table.cols is {table['cols']}",
        )
        group_rows = sum(entry["rows"] for entry in group)
        _require(
            tensor_rows == group_rows,
            f"{where}: source tensor {tensor_name!r} holds {tensor_rows} rows "
            f"but its partitions declare {group_rows}",
        )
        base_row = min(entry["row_offset"] for entry in group)
        cursor = base_row
        for entry in sorted(group, key=lambda item: item["row_offset"]):
            _require(
                entry["row_offset"] == cursor,
                f"{where}: source tensor {tensor_name!r} in {rel} is split by "
                f"non-contiguous partitions at row {entry['row_offset']}",
            )
            cursor += entry["rows"]
        start, end = declared["data_offsets"]
        _require(
            tensor_rows > 0 and (end - start) % tensor_rows == 0,
            f"{where}: source tensor {tensor_name!r} payload is not a whole "
            "number of rows",
        )
        resolved[(shard_name, tensor_name)] = (
            ctx.shard_path(ctx.src_dir, rel),
            start,
            end,
            base_row,
            tensor_rows,
        )

    digest = hashlib.sha256()
    for entry in entries:
        path, start, end, base_row, tensor_rows = resolved[
            (entry["source_shard"], entry["source_tensor"])
        ]
        row_bytes = (end - start) // tensor_rows
        slice_start = start + (entry["row_offset"] - base_row) * row_bytes
        _hash_payload(digest, path, slice_start, slice_start + entry["rows"] * row_bytes)
    return digest.hexdigest(), len(entries)


def _check_source_table_sha256(ctx):
    meta = ctx.manifest()
    digest, count = _source_table_digest(ctx, meta)
    declared = meta["source"]["source_table_sha256"]
    if digest != declared:
        return False, (
            f"source_table_sha256 mismatch over {count} partition slices: "
            f"manifest {declared[:16]}..., recomputed {digest[:16]}..."
        )
    return True, f"source_table_sha256 recomputed over {count} partition slices"


def _tensors_by_name(shards):
    """``{name: (shard, dtype, shape, offsets)}`` plus the ambiguous names."""
    found, ambiguous = {}, set()
    for rel, header in shards.items():
        for name, entry in header.items():
            if name in found:
                ambiguous.add(name)
            else:
                found[name] = (
                    rel,
                    entry["dtype"],
                    tuple(entry["shape"]),
                    tuple(entry["data_offsets"]),
                )
    return found, ambiguous


def _check_non_ple_identity(ctx):
    meta = ctx.manifest()
    ple_names = {entry["source_tensor"] for entry in meta["table"]["partitioning"]}
    src, src_ambiguous = _tensors_by_name(ctx.src_shards())
    dst_shards = ctx.dst_shards()
    dst, dst_ambiguous = _tensors_by_name(dst_shards)
    retained = sorted(set(src) - ple_names)
    for name in retained:
        if name in src_ambiguous:
            return False, (
                f"source tensor {name!r} appears in several source shards, so it "
                "has no single identity"
            )
        if name not in dst:
            return False, f"retained tensor {name!r} is missing from the candidate"
        if name in dst_ambiguous:
            return False, (
                f"candidate tensor {name!r} appears in several candidate shards, so "
                "it has no single identity"
            )
        src_rel, src_dtype, src_shape, src_offsets = src[name]
        dst_rel, dst_dtype, dst_shape, dst_offsets = dst[name]
        if src_dtype != dst_dtype or src_shape != dst_shape:
            return False, (
                f"retained tensor {name!r} identity changed: {src_dtype} "
                f"{list(src_shape)} in {src_rel} -> {dst_dtype} {list(dst_shape)} "
                f"in {dst_rel}"
            )
        src_digest = _sha256_payload(
            ctx.shard_path(ctx.src_dir, src_rel), *src_offsets
        )
        dst_digest = _sha256_payload(
            ctx.shard_path(ctx.dst_dir, dst_rel), *dst_offsets
        )
        if src_digest != dst_digest:
            return False, (
                f"retained tensor {name!r} payload changed: {src_digest[:16]}... in "
                f"{src_rel} -> {dst_digest[:16]}... in {dst_rel}"
            )
    for shard, names in meta["retained_rewrites"].items():
        rel = _resolve_shard(dst_shards, shard, where="retained_rewrites")
        header = dst_shards[rel]
        for name in names:
            if name not in header:
                return False, (
                    f"retained_rewrites declares {name!r} in {shard}, but that shard "
                    "does not contain it"
                )
            if name not in src:
                return False, (
                    f"retained_rewrites declares {name!r}, which is not a source tensor"
                )
    return True, (
        f"{len(retained)} retained non-PLE tensors keep dtype, shape and payload "
        "identity across changed file boundaries"
    )


def _check_index_refs(ctx):
    path = ctx.dst_dir / INDEX_NAME
    if not path.is_file():
        return True, f"no {INDEX_NAME} in the candidate, so nothing references anything"
    try:
        index = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError) as exc:
        return False, f"{INDEX_NAME} unreadable: {exc}"
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return False, f"{INDEX_NAME} has no weight_map object"
    shards = ctx.dst_shards()
    headers = {}
    for name in sorted(weight_map):
        shard = weight_map[name]
        if not isinstance(shard, str) or not shard:
            return False, f"{INDEX_NAME} maps {name!r} to a non-string shard"
        if shard not in shards:
            resolved = [
                rel
                for rel in shards
                if PurePosixPath(rel).name == PurePosixPath(shard).name
            ]
            if len(resolved) != 1:
                return False, f"{INDEX_NAME} maps {name!r} to missing shard {shard}"
            shard = resolved[0]
        if shard not in headers:
            headers[shard] = _read_shard_header(ctx.shard_path(ctx.dst_dir, shard))
        if name not in headers[shard]:
            return False, (
                f"{INDEX_NAME} references nonexistent tensor {name!r} in {shard}"
            )
    return True, f"{len(weight_map)} {INDEX_NAME} entries resolve in existing shards"


def _check_partial_candidate(ctx):
    raw = ctx.raw_manifest()
    if raw is None:
        packed = sorted(
            {
                name
                for header in ctx.dst_shards().values()
                for name in header
                if _PACKED_TENSOR_RE.match(name)
            }
        )
        if packed:
            return False, (
                f"partial candidate: packed PLE tensors without {MANIFEST_NAME}: "
                f"{packed[:3]}"
            )
        return False, (
            f"partial candidate: neither {MANIFEST_NAME} nor packed PLE tensors in "
            f"{ctx.dst_dir}"
        )
    if raw == "unreadable":
        return False, f"{MANIFEST_NAME} is present but unreadable"
    parts = raw.get("parts") if isinstance(raw, dict) else None
    if not isinstance(parts, list) or not parts:
        return False, f"{MANIFEST_NAME} declares no parts; candidate is unfinished"
    missing = []
    for index, part in enumerate(parts):
        name = part.get("file") if isinstance(part, dict) else None
        if not isinstance(name, str) or not name:
            return False, (
                f"{MANIFEST_NAME} parts[{index}] has no file name; candidate is unfinished"
            )
        try:
            path = ctx.part_path(name)
        except VerifyError as exc:
            return False, str(exc)
        if not path.is_file():
            missing.append(name)
    if missing:
        return False, (
            f"partial candidate: {MANIFEST_NAME} names missing part file(s): "
            f"{missing[:3]}; there is no BF16 fallback"
        )
    return True, f"all {len(parts)} {MANIFEST_NAME} part files are present"


def _reference_row(weight_bytes, scale_bytes, cols, global_scale):
    """§1 reconstruction in plain Python; ``(int32 bits, error)``."""
    bits = []
    for col in range(cols):
        byte = weight_bytes[col // 2]
        nibble = (byte & 15) if (col & 1) == 0 else (byte >> 4)
        raw_scale = scale_bytes[col // GROUP_SIZE]
        scale = _e4m3_to_f32(raw_scale)
        if scale is None:
            return None, f"scale byte 0x{raw_scale:02X} is not finite"
        if scale < 0.0:
            return None, f"scale byte 0x{raw_scale:02X} is negative (contract §1)"
        magnitude = _E2M1_MAGNITUDE[nibble & 7]
        quantized = -magnitude if nibble & 8 else magnitude
        # Contract §1 order, matching the loader: (q * scale) * g, then one BF16 RNE.
        value = _f32(quantized * scale)
        value = _f32(value * global_scale)
        bits.append(_bf16_rne_bits(value))
    return bits, None


def _loader_row(weight_bytes, scale_bytes, cols, global_scale):
    """The same §1 reconstruction through torch, i.e. the loader's own path."""
    import torch

    packed = torch.frombuffer(bytearray(weight_bytes), dtype=torch.uint8).to(torch.int32)
    col = torch.arange(cols, dtype=torch.int32)
    fetched = packed[col // 2]
    nibble = torch.where((col & 1) == 0, fetched & 15, fetched >> 4)
    magnitudes = torch.tensor(_E2M1_MAGNITUDE, dtype=torch.float32)
    magnitude = magnitudes[nibble & 7]
    quantized = torch.where((nibble & 8) != 0, -magnitude, magnitude)
    scales = (
        torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    values = (quantized * scales[col // GROUP_SIZE]) * global_scale
    bits = values.contiguous().view(torch.int32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    return bits.tolist()


def _sampled_rows(total, count):
    """Deterministic evenly spread row indices, first and last row included."""
    if count <= 1:
        return [0]
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def _check_dequant_sample(ctx):
    meta = ctx.manifest()
    total = meta["table"]["logical_rows"]
    cols = meta["table"]["cols"]
    if ctx.dequant_sample <= 0:
        return False, "--dequant-sample must be >= 1"
    global_scale = _decode_global_scale(meta)
    if not math.isfinite(global_scale) or global_scale <= 0.0:
        return False, f"global_scale_bits decodes to {global_scale!r}"
    rows = _sampled_rows(total, min(ctx.dequant_sample, total))
    half, scales_per_row = cols // 2, cols // GROUP_SIZE
    for row in rows:
        covering = [
            part
            for part in meta["parts"]
            if part["row_offset"] <= row < part["row_offset"] + part["rows"]
        ]
        _require(len(covering) == 1, f"row {row} is covered by {len(covering)} parts")
        part = covering[0]
        path = ctx.part_path(part["file"])
        _require(path.is_file(), f"part file missing: {part['file']}")
        header = _read_shard_header(path)
        for key in ("weight_tensor", "scale_tensor"):
            _require(
                part[key] in header, f"part {part['file']} lacks {part[key]!r}"
            )
        local = row - part["row_offset"]
        weight_start, _ = header[part["weight_tensor"]]["data_offsets"]
        scale_start, _ = header[part["scale_tensor"]]["data_offsets"]
        weight_bytes = _read_payload(
            path, weight_start + local * half, weight_start + (local + 1) * half
        )
        scale_bytes = _read_payload(
            path,
            scale_start + local * scales_per_row,
            scale_start + (local + 1) * scales_per_row,
        )
        reference, error = _reference_row(weight_bytes, scale_bytes, cols, global_scale)
        if error:
            return False, f"row {row}: {error}"
        loaded = _loader_row(weight_bytes, scale_bytes, cols, global_scale)
        for column, (expected, got) in enumerate(zip(reference, loaded)):
            if expected != got:
                return False, (
                    f"row {row} column {column}: §1 reference decode "
                    f"0x{expected & 0xFFFFFFFF:08X} disagrees with loader decode "
                    f"0x{got & 0xFFFFFFFF:08X}"
                )
    return True, (
        f"{len(rows)} sampled rows decode identically under the §1 reference and "
        "the loader path"
    )


CHECKS = (
    ("manifest_schema", _check_manifest_schema),
    ("global_scale_bits", _check_global_scale_bits),
    ("partitioning", _check_partitioning),
    ("parts_integrity", _check_parts_integrity),
    ("source_table_sha256", _check_source_table_sha256),
    ("non_ple_identity", _check_non_ple_identity),
    ("index_refs", _check_index_refs),
    ("partial_candidate", _check_partial_candidate),
    ("dequant_sample", _check_dequant_sample),
)


def run_verification(src_dir, dst_dir, *, dequant_sample=DEFAULT_DEQUANT_SAMPLE):
    """Run every check in contract order; never raise, always report."""
    ctx = _Ctx(src_dir, dst_dir, dequant_sample)
    checks = []
    for name, check in CHECKS:
        try:
            passed, reason = check(ctx)
        except VerifyError as exc:
            passed, reason = False, str(exc)
        except Exception as exc:  # a crash is a failure, never an implied pass
            passed, reason = False, f"unexpected {type(exc).__name__}: {exc}"
        checks.append(
            {"name": name, "status": "pass" if passed else "fail", "reason": reason}
        )
    return {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "src_dir": str(ctx.src_dir),
        "dst_dir": str(ctx.dst_dir),
        "dequant_sample": dequant_sample,
        "ok": all(check["status"] == "pass" for check in checks),
        "checks": checks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Strict verifier for a packed-NVFP4 PLE candidate checkpoint "
            "(docs/rvn-ple-storage-schema.md)."
        ),
    )
    parser.add_argument("--src-dir", required=True, help="source BF16 checkpoint directory")
    parser.add_argument("--dst-dir", required=True, help="packed candidate directory")
    parser.add_argument("--report", help="write the JSON report to this path")
    parser.add_argument(
        "--dequant-sample",
        type=int,
        default=DEFAULT_DEQUANT_SAMPLE,
        help="rows re-decoded by the dequant check (default %(default)s)",
    )
    args = parser.parse_args(argv)

    report = run_verification(
        args.src_dir, args.dst_dir, dequant_sample=args.dequant_sample
    )
    for check in report["checks"]:
        print(f"{check['status'].upper():4} {check['name']}: {check['reason']}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    passed = sum(1 for check in report["checks"] if check["status"] == "pass")
    print(
        f"verification {'PASSED' if report['ok'] else 'FAILED'}: "
        f"{passed}/{len(report['checks'])} checks"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Strict verifier for a packed-NVFP4 PLE candidate checkpoint.

Conforms to ``docs/rvn-ple-storage-schema.md`` v1. Every rule of that contract
is exactly one named check, evaluated in the fixed order below; the JSON
report lists every check with ``pass``/``fail`` plus a reason, and the exit
code is 0 if and only if every check passes.

    manifest_schema        manifest keys, types and the fixed schema §2 values,
                           including the frozen ``encoder_version`` (§2)
    global_scale_bits      uint32 decodes to a finite positive float, equals the
                           frozen ``g = amax / (6 * 448)``, and that amax is
                           recomputed from the streamed source slices (§1/§2/§4)
    partitioning           numeric SOURCE-ordered, complete, non-overlapping
                           cover; sorted part ids alone are not an order (§2)
    parts_integrity        parts cover the same rows, hold exactly their declared
                           tensor pair, are never reused, their payload digests
                           match the stored bytes, every scale byte is in the §1
                           domain, and first/last_source_tensor is the
                           partitioning entry covering that part's row (§1/§2/§4)
    source_table_sha256    recomputed from the source checkpoint (§4)
    non_ple_identity       every non-PLE tensor keeps dtype, shape and payload
                           digest; a manifest name exempts a tensor only when the
                           candidate no longer carries it (§4)
    index_refs             required above one model shard, and every non-packed
                           candidate tensor must be referenced (§2)
    partial_candidate      no manifest-without-part and no part-without-manifest;
                           there is no BF16 fallback semantics (§2)
    dequant_sample         bounded row re-decode: the §1 reference path and the
                           loader path must agree bit-for-bit (§1)

Quantised-versus-source-BF16 difference is expected and is NOT checked here.
Only stdlib, torch (dequant loader path) and structurally-parsed safetensors
headers are used; no tensor is ever materialised whole for hashing, and only
sampled rows are read for the dequant check.

``--graft`` verifies a different contract instead: the MTP draft head
``tools/rvn_ple/graft_mtp.py`` grafted onto a candidate -- every payload byte
against the MTP source, every base file against the base candidate, plus the
merged index, the stamp and the manifest. Same report shape and exit code; the
nine checks above are not run in that mode (and grafting cannot regress them: it
only adds candidate-side tensors that are byte-identical to source tensors, each
referenced by the merged index).
"""

from __future__ import annotations

import argparse
import bisect
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

# Schema §2 freezes the encoder version string; anything else is a different
# encoder and must not be blessed by this gate (the loader fails closed too).
ENCODER_VERSION = "rvn-ple-nvfp4-r1"
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
# Bytes per element for every safetensors store code the verifier accepts
# (schema §4 makes dtype+shape part of the digest input, so a declared payload
# range must be exactly ``itemsize * prod(shape)``). This is the complete store
# set safetensors 0.8 parses (probe: every other code it rejects itself, so
# failing closed on an unknown one cannot refuse a checkpoint the loader reads).
# ``None`` marks the sub-byte code: two packed values per element, so its extent
# is not derivable from numel and only the exact-extent rule skips it.
_STORE_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
    "F4": None,
}

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
    ranges = []
    file_size = path.stat().st_size
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
        _require(
            # Unknown codes fail closed (the safetensors parser rejects them too);
            # the one sub-byte code carries itemsize None on purpose.
            entry["dtype"] in _STORE_ITEMSIZE,
            f"unreadable shard (unsupported dtype {entry['dtype']!r} for "
            f"{name!r}): {path}",
        )
        itemsize = _STORE_ITEMSIZE[entry["dtype"]]
        numel = 1
        for dim in entry["shape"]:
            numel *= dim
        offsets = entry.get("data_offsets")
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(off, int) and off >= 0 for off in offsets)
            and offsets[0] <= offsets[1],
            f"unreadable shard (bad data_offsets for {name!r}): {path}",
        )
        # Schema §4 fixes the digest over the raw payload of THIS dtype+shape, so
        # the declared range must be exactly the tensor the header describes --
        # whenever the store code's width is per-element (skipped for F4, whose
        # extent is not derivable from numel; bounds and disjointness still hold).
        if itemsize is not None:
            _require(
                offsets[1] - offsets[0] == itemsize * numel,
                f"shard {path} tensor {name!r} is {entry['dtype']} {entry['shape']} "
                f"= {itemsize * numel} bytes, but data_offsets span "
                f"{offsets[1] - offsets[0]} bytes",
            )
        _require(
            payload_base + offsets[1] <= file_size,
            f"shard {path} tensor {name!r} payload ends at byte "
            f"{payload_base + offsets[1]}, past the {file_size}-byte file",
        )
        if offsets[1] > offsets[0]:  # zero-length tensors may share an offset
            ranges.append((offsets[0], offsets[1], name))
        tensors[name] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [offsets[0] + payload_base, offsets[1] + payload_base],
        }
    # Header key order is writer-defined (HF merge tools and convert's own
    # header writer emit arbitrary orders), so pair up by offset, not by key.
    ranges.sort()
    for (_, previous_end, previous), (start, _, name) in zip(ranges, ranges[1:]):
        _require(
            start >= previous_end,
            f"shard {path}: payload of {name!r} overlaps that of {previous!r}",
        )
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


def _hash_payload(digest, path: Path, start, end, watch=None):
    """Feed ``[start, end)`` of a shard's stored payload into ``digest``.

    ``watch`` (optional) sees every raw block through ``watch.feed(block)``, so
    the value-domain rules the loader enforces over ALL bytes -- the schema §1
    scale domain and the source amax -- are checked in this same streaming pass
    rather than only on the rows the dequant check samples.
    """
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
            if watch is not None:
                watch.feed(block)
            remaining -= len(block)


def _sha256_payload(path: Path, start, end, watch=None) -> str:
    digest = hashlib.sha256()
    _hash_payload(digest, path, start, end, watch=watch)
    return digest.hexdigest()


# Schema §1: block scales are finite and non-negative on read, i.e. no sign bit
# (any byte >= 0x80) and no NaN encoding (0x7F). The loader rejects a whole part
# on any such byte, so the gate has to see every scale byte, not sampled rows.
_BAD_SCALE_BYTE = re.compile(rb"[\x7f\x80-\xff]")


class _ScaleDomain:
    """Watch remembering the first scale byte outside the schema §1 domain."""

    def __init__(self):
        self.bad = None

    def feed(self, block):
        if self.bad is None:
            found = _BAD_SCALE_BYTE.search(block)
            if found is not None:
                self.bad = found.group()[0]


class _SourceAmax:
    """Watch recomputing ``amax`` over streamed BF16 source payload bytes.

    This is the same quantity the converter scans and the loader's
    ``g = amax / (6 * 448)`` is frozen against: float32 of the stored BF16
    bits, maximum absolute value, NaN reported rather than skipped.
    """

    def __init__(self):
        self.value = 0.0
        self._odd = b""

    def feed(self, block):
        import torch

        if self._odd:
            block = self._odd + block
            self._odd = b""
        if len(block) % 2:
            self._odd, block = block[-1:], block[:-1]
        if not block:
            return
        current = float(
            torch.frombuffer(bytearray(block), dtype=torch.uint16)
            .view(torch.bfloat16)
            .to(torch.float32)
            .abs()
            .max()
        )
        if math.isnan(current) or current > self.value:
            self.value = current


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
        # The loader rejects a negative amax (patches/0049 parse_manifest), so a
        # gate that accepted one would report PASSED for a checkpoint that then
        # hard-errors at boot.
        _is_num(source.get("amax"))
        and math.isfinite(source["amax"])
        and source["amax"] >= 0,
        "source.amax must be a finite non-negative number",
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
        meta.get("encoder_version") == ENCODER_VERSION,
        f"encoder_version must be {ENCODER_VERSION!r} (schema §2), got "
        f"{meta.get('encoder_version')!r}",
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


def _expected_global_scale_bits(amax):
    """Schema §1 bits of ``g = amax / (6 * 448)``; all-zero table -> ``g = 1.0``.

    Deliberately a copy of the loader's ``expected_global_scale_bits``
    (patches/0049-rvn-ple-packed-loader.patch), because the loader is what this
    gate exists to pre-empt: a manifest whose bits are not this function of its
    amax rescales every decoded PLE value.
    """
    _require(
        _is_num(amax) and math.isfinite(amax) and amax >= 0,
        f"source.amax must be finite and non-negative, got {amax!r}",
    )
    g = 1.0 if amax == 0.0 else _f32(amax / (6.0 * 448.0))
    return struct.unpack("<I", struct.pack("<f", g))[0]


def _natural_key(name):
    """Numeric-aware sort key: digit runs compare as integers (2 < 10)."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", name)
    )


class _Ctx:
    """Lazy shared state for the checks; bad input always raises VerifyError."""

    def __init__(self, src_dir, dst_dir, dequant_sample):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.dequant_sample = dequant_sample
        self._manifest = None
        self._src_shards = None
        self._dst_shards = None
        self._source_scan = None

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

    def source_scan(self):
        """One streaming pass over the source slices: ``(§4 digest, count, amax)``.

        Cached, so the global-scale check and ``source_table_sha256`` share the
        same read: the amax is recomputed from the very bytes the digest covers,
        which is what stops ``source.amax`` from being self-attested.
        """
        if self._source_scan is None:
            self._source_scan = _source_table_scan(self, self.manifest())
        return self._source_scan


def _check_manifest_schema(ctx):
    ctx.manifest()
    return True, "manifest keys, types and fixed values conform to contract §2"


def _check_global_scale_bits(ctx):
    meta = ctx.manifest()
    bits = meta["encoding"]["global_scale_bits"]
    global_scale = _decode_global_scale(meta)
    if not math.isfinite(global_scale):
        return False, f"global_scale_bits decodes to non-finite {global_scale!r}"
    if global_scale <= 0.0:
        return False, f"global_scale_bits decodes to non-positive {global_scale!r}"
    amax = meta["source"]["amax"]
    expected = _expected_global_scale_bits(amax)
    if bits != expected:
        return False, (
            f"conflicting global scale bits: encoding.global_scale_bits "
            f"{bits:#010x} is not source.amax {amax!r} / (6 * 448) (expected "
            f"{expected:#010x}); the loader rejects this manifest, and every "
            "decoded PLE value would carry the wrong scale"
        )
    _, count, scanned = ctx.source_scan()
    if not math.isfinite(scanned):
        return False, (
            f"source amax recomputed over {count} partition slices is "
            f"{scanned!r}: the source table holds a non-finite value"
        )
    if scanned != float(amax):
        return False, (
            f"source.amax is self-attested: the manifest declares {amax!r} but "
            f"the {count} streamed source slices hold {scanned!r}, so global "
            f"scale {global_scale!r} rescales every decoded PLE value"
        )
    return True, (
        f"global_scale_bits {bits:#010x} = amax/(6*448) for the recomputed "
        f"source amax {scanned!r} over {count} partition slices"
    )


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
    # Contract §2: partitioning follows NUMERIC SOURCE order. Sorted part ids
    # are self-attested, so a converter or tamper that orders partitions
    # lexicographically by shard/tensor name and then renumbers part = 0..n-1
    # with contiguous row_offsets would pass every other check while shipping a
    # table whose rows are in layer-10-before-layer-2 order. Derive the order
    # from the source itself: shard position, then numeric tensor name.
    order = [
        (_natural_key(entry["source_shard"]), _natural_key(entry["source_tensor"]))
        for entry in entries
    ]
    for index in range(1, len(order)):
        if order[index] < order[index - 1]:
            before, after = entries[index - 1], entries[index]
            return False, (
                "partitioning is not in numeric source order (contract §2 "
                "forbids lexicographic order): part "
                f"{after['part']} of {after['source_shard']}/"
                f"{after['source_tensor']} sits at rows "
                f"{after['row_offset']}..{after['row_offset'] + after['rows'] - 1} "
                f"before part {before['part']} of {before['source_shard']}/"
                f"{before['source_tensor']}, which is later in the source"
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


def _partition_at(partitions, starts, row):
    """The partitioning entry covering ``row``, or ``None`` when none does.

    ``partitions`` is sorted by ``row_offset`` and ``starts`` its offsets; the
    table can hold tens of millions of rows, so this is a lookup, never a
    per-row map.
    """
    index = bisect.bisect_right(starts, row) - 1
    if index < 0:
        return None
    entry = partitions[index]
    if row >= entry["row_offset"] + entry["rows"]:
        return None
    return entry


def _check_parts_integrity(ctx):
    meta = ctx.manifest()
    cols = meta["table"]["cols"]
    total = meta["table"]["logical_rows"]
    wanted = {
        "weight_tensor": (WEIGHT_STORE_DTYPE, (0, cols // 2)),
        "scale_tensor": (SCALE_STORE_DTYPE, (0, cols // GROUP_SIZE)),
    }
    partitions = sorted(
        meta["table"]["partitioning"], key=lambda entry: entry["row_offset"]
    )
    starts = [entry["row_offset"] for entry in partitions]
    spans = []
    declared_parts = set()
    for part in meta["parts"]:
        path = ctx.part_path(part["file"])
        _require(path.is_file(), f"part file missing: {part['file']}")
        if part["rows"] <= 0:
            return False, f"part {part['file']} has rows <= 0"
        identity = (part["file"], part["weight_tensor"], part["scale_tensor"])
        if identity in declared_parts:
            return False, (
                f"parts[] repeats {part['file']} with tensor pair "
                f"{part['weight_tensor']!r}/{part['scale_tensor']!r} at another "
                "row_offset: the loader would write those PLE rows twice"
            )
        declared_parts.add(identity)
        header = _read_shard_header(path)
        expected_names = {part["weight_tensor"], part["scale_tensor"]}
        if set(header) != expected_names:
            return False, (
                f"part {part['file']} must hold exactly "
                f"{sorted(expected_names)} (contract §1 pair); its header names "
                f"{sorted(header)}"
            )
        for key, sha_key in (
            ("weight_tensor", "sha256_weights"),
            ("scale_tensor", "sha256_scales"),
        ):
            name = part[key]
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
            # The scale payload is hashed anyway, so the loader's whole-part
            # scale-domain rule is enforced here over every byte, not only the
            # rows the dequant check samples.
            watch = _ScaleDomain() if key == "scale_tensor" else None
            digest = _sha256_payload(path, start, end, watch=watch)
            if digest != part[sha_key]:
                return False, (
                    f"sha256 {sha_key} mismatch for {part['file']}:{name}: "
                    f"manifest {part[sha_key][:16]}..., stored payload "
                    f"{digest[:16]}..."
                )
            if watch is not None and watch.bad is not None:
                return False, (
                    f"part {part['file']} scale byte 0x{watch.bad:02X} is outside "
                    "the schema §1 domain (scales must be finite and "
                    "non-negative); the loader rejects this whole part"
                )
        for key, row in (
            ("first_source_tensor", part["row_offset"]),
            ("last_source_tensor", part["row_offset"] + part["rows"] - 1),
        ):
            entry = _partition_at(partitions, starts, row)
            if entry is None:
                return False, (
                    f"part {part['file']} declares {key} for row {row}, which no "
                    "partitioning entry assigns to a source tensor"
                )
            if part[key] != entry["source_tensor"]:
                return False, (
                    f"part {part['file']} declares {key} {part[key]!r}, but "
                    f"partitioning assigns row {row} to "
                    f"{entry['source_shard']}/{entry['source_tensor']}: these "
                    "packed bytes are not bound to the rows the loader writes "
                    "them into"
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
        "payload digests, exact two-tensor headers, in-domain scale bytes and "
        "source tensors bound to their row ranges"
    )


def _source_table_scan(ctx, meta):
    """§4 digest, partition count and recomputed amax over the source slices.

    Entries splitting one source tensor must tile that tensor's rows, so the
    ascending-part concat covers the whole logical table exactly once. The same
    streamed bytes yield the amax, so no manifest scale stays self-attested.
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
    amax = _SourceAmax()
    for entry in entries:
        path, start, end, base_row, tensor_rows = resolved[
            (entry["source_shard"], entry["source_tensor"])
        ]
        row_bytes = (end - start) // tensor_rows
        slice_start = start + (entry["row_offset"] - base_row) * row_bytes
        _hash_payload(
            digest,
            path,
            slice_start,
            slice_start + entry["rows"] * row_bytes,
            watch=amax,
        )
    return digest.hexdigest(), len(entries), amax.value


def _check_source_table_sha256(ctx):
    meta = ctx.manifest()
    # The cached streaming pass (also used by global_scale_bits) covers exactly
    # these slices, so the digest is recomputed once per run, not once per check.
    digest, count, _ = ctx.source_scan()
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
    src, src_ambiguous = _tensors_by_name(ctx.src_shards())
    dst_shards = ctx.dst_shards()
    dst, dst_ambiguous = _tensors_by_name(dst_shards)
    # A manifest name counts as packed only when the candidate no longer carries
    # it. Otherwise naming an ordinary weight (a gate_proj, say) in
    # table.partitioning would drop that weight from the dtype/shape/payload
    # comparison the operator relies on, so any name present on both sides is
    # compared in full.
    packed = {
        entry["source_tensor"]
        for entry in meta["table"]["partitioning"]
        if entry["source_tensor"] not in dst
    }
    retained = sorted(set(src) - packed)
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


def _declared_parts(ctx):
    """``(part files, part tensor names)`` from the raw manifest, if readable."""
    raw = ctx.raw_manifest()
    files, names = set(), set()
    parts = raw.get("parts") if isinstance(raw, dict) else None
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("file"), str):
                # _resolve_shard tolerates a directory prefix in a declared
                # shard name, so classification must accept both spellings of a
                # part file or a declared part counts as a model shard.
                declared = PurePosixPath(part["file"])
                files.add(declared.as_posix())
                files.add(declared.name)
            for key in ("weight_tensor", "scale_tensor"):
                if isinstance(part.get(key), str):
                    names.add(part[key])
    return files, names


def _check_index_refs(ctx):
    files, part_names = _declared_parts(ctx)
    shards = ctx.dst_shards()
    # Manifest part files are packed payloads, not model shards; anything else in
    # the candidate tree is a model shard the index has to account for.
    model_shards = {
        rel
        for rel in shards
        if rel not in files and PurePosixPath(rel).name not in files
    }
    path = ctx.dst_dir / INDEX_NAME
    if not path.is_file():
        if len(model_shards) > 1:
            return False, (
                f"no {INDEX_NAME} in the candidate although it holds "
                f"{len(model_shards)} model shards: nothing says which shard owns "
                "each tensor, and a missing index must not be a green reference "
                "check"
            )
        return True, (
            f"no {INDEX_NAME} needed: the candidate holds one model shard"
        )
    try:
        index = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError) as exc:
        return False, f"{INDEX_NAME} unreadable: {exc}"
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return False, f"{INDEX_NAME} has no weight_map object"
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
    referenced = set(weight_map)
    unreferenced = sorted(
        name
        for rel in sorted(model_shards)
        for name in shards[rel]
        if name not in part_names and name not in referenced
    )
    if unreferenced:
        return False, (
            f"{INDEX_NAME} does not reference candidate tensor(s) "
            f"{unreferenced[:3]}{'...' if len(unreferenced) > 3 else ''}: a "
            "leftover or injected shard must not ride along behind a clean report"
        )
    covered = sum(
        1 for rel in model_shards for name in shards[rel] if name not in part_names
    )
    return True, (
        f"{len(weight_map)} {INDEX_NAME} entries resolve in existing shards and "
        f"all {covered} non-packed candidate tensors are referenced"
    )


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


# --------------------------------------------------------------------------
# Graft mode (tools/rvn_ple/graft_mtp.py)
#
# A different contract from the packed-PLE schema above. The grafted candidate
# keeps every base file hardlinked (so the target shards must stay byte-identical
# to the base), and adds one draft-head shard whose payload bytes must stay
# byte-identical to the MTP source they were copied from -- no requantise,
# because NEXTN is lossless w.r.t. the target only while the draft is still the
# draft its author trained. Each rule below is one named check, with the same
# report shape and exit code as the PLE checks above.
# --------------------------------------------------------------------------

GRAFT_MANIFEST_NAME = "mtp_graft.json"
GRAFT_STAMP_KEY = "rvn_mtp_graft"
CONFIG_NAME = "config.json"
MTP_PREFIX = "mtp."
# Frozen with the loader rule: the ``mtp`` weight-name exemption applies only to a
# checkpoint whose stamp names this encoder version and this count, so verifying
# a graft means verifying those literals, not just the bytes.
GRAFT_ENCODER_VERSION = "rvn-mtp-graft-r1"
GRAFT_COUNT = 1
GRAFT_MTP_NUM_HIDDEN_LAYERS = 1


def _validate_graft_manifest(meta):
    _require(isinstance(meta, dict), f"{GRAFT_MANIFEST_NAME} is not a JSON object")
    _require(
        set(meta) == {"source", "tensors"},
        f"{GRAFT_MANIFEST_NAME} must hold exactly source+tensors, got "
        f"{sorted(meta)}",
    )
    _require(
        isinstance(meta["source"], str) and meta["source"] != "",
        f"{GRAFT_MANIFEST_NAME} source must be a non-empty string",
    )
    tensors = meta["tensors"]
    _require(
        isinstance(tensors, dict) and tensors,
        f"{GRAFT_MANIFEST_NAME} tensors must be a non-empty object",
    )
    for name, entry in tensors.items():
        _require(
            isinstance(entry, dict)
            and set(entry) == {"file", "sha256", "dtype", "shape"},
            f"{GRAFT_MANIFEST_NAME} entry {name!r} must hold exactly "
            f"file/sha256/dtype/shape, got {sorted(entry) if isinstance(entry, dict) else type(entry).__name__}",
        )
        _require(
            isinstance(entry["file"], str) and entry["file"] != "",
            f"{GRAFT_MANIFEST_NAME} entry {name!r} has no source shard file",
        )
        _require(
            _is_hex64(entry["sha256"]),
            f"{GRAFT_MANIFEST_NAME} entry {name!r} sha256 is not 64 hex chars",
        )
        _require(
            isinstance(entry["dtype"], str) and entry["dtype"] != "",
            f"{GRAFT_MANIFEST_NAME} entry {name!r} has no dtype",
        )
        _require(
            isinstance(entry["shape"], list)
            and all(_is_int(dim) and dim >= 0 for dim in entry["shape"]),
            f"{GRAFT_MANIFEST_NAME} entry {name!r} has a bad shape",
        )


def _read_config(root: Path, *, where):
    path = root / CONFIG_NAME
    _require(path.is_file(), f"missing {CONFIG_NAME} in {root}")
    try:
        config = json.loads(path.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError) as exc:
        raise VerifyError(f"{where} {CONFIG_NAME} unreadable: {exc}") from exc
    _require(isinstance(config, dict), f"{where} {CONFIG_NAME} is not an object")
    return config


class _GraftCtx:
    """Lazy shared state for the graft checks; bad input raises VerifyError."""

    def __init__(self, source_dir, base_dir, out_dir):
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.out_dir = Path(out_dir)
        for name, root in (("source", self.source_dir), ("base", self.base_dir),
                           ("out", self.out_dir)):
            _require(root.is_dir(), f"missing {name} directory: {root}")
        self._manifest = None
        self._indices = {}
        self._configs = {}
        self._source_headers = {}
        self._graft_headers = None

    def shard_path(self, root, rel_name):
        return root.joinpath(*PurePosixPath(rel_name).parts)

    def manifest(self):
        if self._manifest is None:
            path = self.out_dir / GRAFT_MANIFEST_NAME
            _require(path.is_file(), f"missing {GRAFT_MANIFEST_NAME} in {self.out_dir}")
            try:
                meta = json.loads(
                    path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
                )
            except (OSError, ValueError) as exc:
                raise VerifyError(f"unreadable {GRAFT_MANIFEST_NAME}: {exc}") from exc
            _validate_graft_manifest(meta)
            self._manifest = meta
        return self._manifest

    def weight_map(self, which):
        """``{name: shard}`` of the source/base/out index, read once."""
        if which not in self._indices:
            root = getattr(self, f"{which}_dir")
            path = root / INDEX_NAME
            _require(path.is_file(), f"missing {INDEX_NAME} in {root}")
            try:
                index = json.loads(
                    path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
                )
            except (OSError, ValueError) as exc:
                raise VerifyError(f"{which} {INDEX_NAME} unreadable: {exc}") from exc
            _require(isinstance(index, dict), f"{which} {INDEX_NAME} is not an object")
            weight_map = index.get("weight_map")
            _require(
                isinstance(weight_map, dict) and weight_map,
                f"{which} {INDEX_NAME} has no weight_map object",
            )
            for name, shard in weight_map.items():
                _require(
                    isinstance(shard, str) and shard != "",
                    f"{which} {INDEX_NAME} maps {name!r} to a non-string shard",
                )
            self._indices[which] = weight_map
        return self._indices[which]

    def config(self, which):
        if which not in self._configs:
            self._configs[which] = _read_config(
                getattr(self, f"{which}_dir"), where=which
            )
        return self._configs[which]

    def source_header(self, rel_name):
        """Parsed header of one SOURCE shard, read once per shard."""
        if rel_name not in self._source_headers:
            self._source_headers[rel_name] = _read_shard_header(
                self.shard_path(self.source_dir, rel_name)
            )
        return self._source_headers[rel_name]

    def graft_headers(self):
        """Parsed headers of the grafted shard files named by the out index.

        Only the grafted shards are parsed here: the base side of the contract is
        the hardlink check, and re-parsing 369 target headers to prove a link is
        the same inode would be a slower way of saying ``st_ino``.
        """
        if self._graft_headers is None:
            base_shards = set(self.weight_map("base").values())
            weight_map = self.weight_map("out")
            headers = {}
            for name in sorted(self.manifest()["tensors"]):
                shard = weight_map.get(name)
                _require(
                    shard is not None,
                    f"{INDEX_NAME} has no entry for grafted tensor {name!r}",
                )
                _require(
                    shard not in base_shards,
                    f"grafted tensor {name!r} is indexed into base shard {shard}: "
                    "the draft head must live in a shard of its own, never in a "
                    "hardlinked target shard",
                )
                if shard not in headers:
                    headers[shard] = _read_shard_header(
                        self.shard_path(self.out_dir, shard)
                    )
            self._graft_headers = headers
        return self._graft_headers


def _check_graft_manifest_schema(ctx):
    tensors = ctx.manifest()["tensors"]
    return True, (
        f"{GRAFT_MANIFEST_NAME} declares {len(tensors)} tensors with "
        "file/sha256/dtype/shape and a source path"
    )


def _check_graft_completeness(ctx):
    """Nothing dropped, nothing invented: manifest == source mtp set."""
    declared = set(ctx.manifest()["tensors"])
    source_mtp = {
        name for name in ctx.weight_map("source") if name.startswith(MTP_PREFIX)
    }
    missing = sorted(source_mtp - declared, key=_natural_key)
    _require(
        not missing,
        f"graft is missing {len(missing)} source {MTP_PREFIX}* tensor(s), starting "
        f"with {missing[:3]}: a partial draft head would load and then mispredict",
    )
    extra = sorted(declared - source_mtp, key=_natural_key)
    _require(
        not extra,
        f"{GRAFT_MANIFEST_NAME} declares {len(extra)} tensor(s) the source index "
        f"does not hold, starting with {extra[:3]}",
    )
    return True, (
        f"all {len(declared)} source {MTP_PREFIX}* tensors are grafted and nothing "
        "else is claimed"
    )


def _check_graft_provenance(ctx):
    """The manifest must describe the checkpoint actually being verified."""
    meta = ctx.manifest()
    _require(
        meta["source"] == str(ctx.source_dir.resolve()),
        f"{GRAFT_MANIFEST_NAME} names source {meta['source']!r}, but --source is "
        f"{str(ctx.source_dir.resolve())!r}: the draft head must be checked against "
        "the checkpoint it was copied from",
    )
    for name in sorted(meta["tensors"], key=_natural_key):
        entry = meta["tensors"][name]
        header = ctx.source_header(entry["file"])
        src = header.get(name)
        _require(
            src is not None,
            f"{GRAFT_MANIFEST_NAME} says {name!r} lives in {entry['file']}, which "
            "does not contain it",
        )
        _require(
            src["dtype"] == entry["dtype"] and src["shape"] == entry["shape"],
            f"{GRAFT_MANIFEST_NAME} declares {name!r} as {entry['dtype']} "
            f"{entry['shape']}, but the source shard holds {src['dtype']} "
            f"{src['shape']}",
        )
    return True, (
        f"source provenance resolves for every tensor across "
        f"{len(ctx._source_headers)} source shard(s)"
    )


def _check_graft_payload_identity(ctx):
    """Per tensor: recorded sha256 == source bytes == grafted bytes."""
    tensors = ctx.manifest()["tensors"]
    graft_headers = ctx.graft_headers()
    weight_map = ctx.weight_map("out")
    payload_bytes = 0
    for name in sorted(tensors, key=_natural_key):
        entry = tensors[name]
        src = ctx.source_header(entry["file"])[name]
        shard = weight_map[name]
        dst = graft_headers[shard].get(name)
        _require(
            dst is not None,
            f"{INDEX_NAME} maps grafted {name!r} to {shard}, which does not hold it",
        )
        _require(
            dst["dtype"] == entry["dtype"] == src["dtype"]
            and dst["shape"] == entry["shape"] == src["shape"],
            f"grafted tensor {name!r} is {dst['dtype']} {dst['shape']} in {shard}, "
            f"declared {entry['dtype']} {entry['shape']}, source {src['dtype']} "
            f"{src['shape']}",
        )
        want = entry["sha256"]
        source_digest = _sha256_payload(
            ctx.shard_path(ctx.source_dir, entry["file"]), *src["data_offsets"]
        )
        _require(
            source_digest == want,
            f"grafted tensor {name!r} records {want[:16]}... but its SOURCE payload "
            f"in {entry['file']} is {source_digest[:16]}...: the manifest was not "
            "written from these bytes",
        )
        landed = _sha256_payload(
            ctx.shard_path(ctx.out_dir, shard), *dst["data_offsets"]
        )
        _require(
            landed == want,
            f"grafted tensor {name!r} payload changed: {want[:16]}... in "
            f"{entry['file']} -> {landed[:16]}... in {shard}",
        )
        payload_bytes += dst["data_offsets"][1] - dst["data_offsets"][0]
    return True, (
        f"{len(tensors)} grafted payloads ({payload_bytes} bytes) are byte-identical "
        "to the source bytes they were copied from"
    )


def _check_graft_index(ctx):
    base_map = ctx.weight_map("base")
    out_map = ctx.weight_map("out")
    grafted = set(ctx.manifest()["tensors"])
    with_mtp = sorted((n for n in base_map if "mtp" in n), key=_natural_key)
    _require(
        not with_mtp,
        f"base index already carries {len(with_mtp)} tensor name(s) containing "
        f"'mtp' (e.g. {with_mtp[:3]}): the graft rule is additive, so a base "
        "draft head would make base+grafted ambiguous",
    )
    clash = sorted(base_map.keys() & grafted, key=_natural_key)
    _require(
        not clash,
        f"graft declares {len(clash)} tensor name(s) the base already owns, "
        f"starting with {clash[:3]}",
    )
    dropped = sorted(base_map.keys() - out_map.keys(), key=_natural_key)
    _require(
        not dropped,
        f"{INDEX_NAME} lost {len(dropped)} base tensor(s), starting with "
        f"{dropped[:3]}",
    )
    foreign = sorted(out_map.keys() - base_map.keys() - grafted, key=_natural_key)
    _require(
        not foreign,
        f"{INDEX_NAME} references {len(foreign)} tensor(s) that are neither base "
        f"nor grafted, starting with {foreign[:3]}: a leftover or injected shard "
        "must not ride along behind a clean report",
    )
    moved = sorted(
        (n for n in base_map if out_map[n] != base_map[n]), key=_natural_key
    )
    _require(
        not moved,
        f"{INDEX_NAME} remaps {len(moved)} base tensor(s) to another shard, "
        f"starting with {moved[:3]}",
    )
    shards = sorted({out_map[name] for name in grafted})
    for shard in shards:
        path = ctx.shard_path(ctx.out_dir, shard)
        _require(path.is_file(), f"{INDEX_NAME} names grafted shard {shard} which is absent")
        # A grafted shard is the only file in the graft that may be rewritten by
        # a repair path; a hardlink there would let one corrupt the shipped base.
        _require(
            path.stat().st_nlink == 1,
            f"grafted shard {shard} is shared with another file "
            f"(st_nlink={path.stat().st_nlink}): it must be a private file",
        )
        stored = set(ctx.graft_headers()[shard]) - set(grafted)
        _require(
            not stored,
            f"grafted shard {shard} stores {len(stored)} tensor(s) "
            f"{sorted(stored, key=_natural_key)[:3]} that {GRAFT_MANIFEST_NAME} does "
            "not declare: those bytes came from nowhere this gate can vouch for",
        )
    return True, (
        f"{len(out_map)} {INDEX_NAME} entries = {len(base_map)} base (unmoved) + "
        f"{len(grafted)} grafted in {len(shards)} private shard(s)"
    )


def _check_graft_config_stamp(ctx):
    base_config = ctx.config("base")
    out_config = ctx.config("out")
    stamp = out_config.get(GRAFT_STAMP_KEY)
    _require(
        isinstance(stamp, dict),
        f"{CONFIG_NAME} has no {GRAFT_STAMP_KEY} object: without the stamp the "
        "loader rule keeps rejecting every mtp weight name",
    )
    _require(
        stamp.get("encoder_version") == GRAFT_ENCODER_VERSION,
        f"{GRAFT_STAMP_KEY}.encoder_version is {stamp.get('encoder_version')!r}, "
        f"expected {GRAFT_ENCODER_VERSION!r}",
    )
    _require(
        stamp.get("count") == GRAFT_COUNT and _is_int(stamp.get("count")),
        f"{GRAFT_STAMP_KEY}.count is {stamp.get('count')!r}, expected "
        f"{GRAFT_COUNT}",
    )
    _require(
        stamp.get("source") == ctx.manifest()["source"],
        f"{GRAFT_STAMP_KEY}.source is {stamp.get('source')!r} but "
        f"{GRAFT_MANIFEST_NAME} names {ctx.manifest()['source']!r}",
    )
    declared = int(out_config.get("mtp_num_hidden_layers", 0) or 0) + int(
        out_config.get("num_nextn_predict_layers", 0) or 0
    )
    _require(
        declared == GRAFT_COUNT,
        f"grafted {CONFIG_NAME} declares {declared} MTP layer(s) "
        f"(mtp_num_hidden_layers + num_nextn_predict_layers), expected "
        f"{GRAFT_COUNT}: the loader exemption is keyed on this sum",
    )
    # The contract equation: grafted config == base config with the stamp and
    # mtp_num_hidden_layers=1 applied, and NOTHING else touched. The stamp itself
    # is validated field by field above, so it is compared out of the way.
    _require(
        GRAFT_STAMP_KEY not in base_config,
        f"base {CONFIG_NAME} already carries a {GRAFT_STAMP_KEY} stamp: the graft "
        "rule is defined for the ungrafted candidate only",
    )
    want = dict(base_config, mtp_num_hidden_layers=GRAFT_MTP_NUM_HIDDEN_LAYERS)
    got = {
        key: value for key, value in out_config.items() if key != GRAFT_STAMP_KEY
    }
    changed = sorted(
        key
        for key in set(got) | set(want)
        if got.get(key, "<absent>") != want.get(key, "<absent>")
    )
    _require(
        not changed,
        f"grafted {CONFIG_NAME} changed base field(s) {changed[:3]} beyond the "
        f"frozen {GRAFT_STAMP_KEY} stamp and mtp_num_hidden_layers="
        f"{GRAFT_MTP_NUM_HIDDEN_LAYERS}",
    )
    return True, (
        f"{CONFIG_NAME} is the base config with mtp_num_hidden_layers="
        f"{GRAFT_MTP_NUM_HIDDEN_LAYERS} and the {GRAFT_ENCODER_VERSION} stamp"
    )


def _same_graft_file(src: Path, dst: Path):
    """True iff ``dst`` is the base file unchanged: same inode, else full compare."""
    ss, ds = src.stat(), dst.stat()
    if (ss.st_dev, ss.st_ino) == (ds.st_dev, ds.st_ino):
        return True
    if ss.st_size != ds.st_size:
        return False
    with open(src, "rb") as a, open(dst, "rb") as b:
        while True:
            block = a.read(CHUNK)
            if not block:
                return True
            if block != b.read(len(block)):
                return False


def _check_graft_files(ctx):
    """Every base file the graft promised is present, and nothing extra rides along.

    "Nothing extra" matters more than it looks: safetensors weight iterators walk
    every shard file in the directory, so an undeclared tensor in any of them still
    reaches the loader even though no index entry names it.
    """
    checked = linked = 0
    base_shards = set()
    for path in sorted(ctx.base_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rel = path.relative_to(ctx.base_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.suffix == ".safetensors":
            base_shards.add(rel.as_posix())
        if len(rel.parts) == 1 and rel.name in (
            CONFIG_NAME,
            INDEX_NAME,
            GRAFT_MANIFEST_NAME,
        ):
            continue  # these three are the graft's own outputs
        out = ctx.out_dir.joinpath(*rel.parts)
        _require(out.is_file(), f"graft is missing base file {rel.as_posix()}")
        _require(
            _same_graft_file(path, out),
            f"graft file {rel.as_posix()} is not the base file unchanged",
        )
        checked += 1
        sbase, sout = path.stat(), out.stat()
        if (sbase.st_dev, sbase.st_ino) == (sout.st_dev, sout.st_ino):
            linked += 1
    _require(checked > 0, f"no base files found under {ctx.base_dir} to compare")
    grafted_shards = set(ctx.graft_headers())
    for path in sorted(ctx.out_dir.rglob("*.safetensors")):
        rel = path.relative_to(ctx.out_dir).as_posix()
        _require(
            rel in base_shards or rel in grafted_shards,
            f"graft carries shard {rel} which is neither a base shard nor a shard "
            f"{GRAFT_MANIFEST_NAME} declares: weight iterators read every shard file "
            "in the directory, so its tensors would still reach the loader",
        )
    return True, (
        f"{checked} base files present unchanged in the graft ({linked} hardlinked "
        "to the base inode, so they cannot have drifted), and no extra shard "
        f"besides the {len(grafted_shards)} grafted one(s)"
    )


GRAFT_CHECKS = (
    ("graft_manifest_schema", _check_graft_manifest_schema),
    ("graft_completeness", _check_graft_completeness),
    ("graft_provenance", _check_graft_provenance),
    ("graft_payload_identity", _check_graft_payload_identity),
    ("graft_index", _check_graft_index),
    ("graft_config_stamp", _check_graft_config_stamp),
    ("graft_files", _check_graft_files),
)


def run_graft_verification(source, base, out):
    """Run every graft check in contract order; never raise, always report."""
    try:
        ctx = _GraftCtx(source, base, out)
    except VerifyError as exc:
        return {
            "tool": TOOL_NAME,
            "version": TOOL_VERSION,
            "mode": "graft",
            "source": str(source),
            "base": str(base),
            "out": str(out),
            "ok": False,
            "checks": [{"name": "graft_inputs", "status": "fail", "reason": str(exc)}],
        }
    checks = []
    for name, check in GRAFT_CHECKS:
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
        "mode": "graft",
        "source": str(ctx.source_dir),
        "base": str(ctx.base_dir),
        "out": str(ctx.out_dir),
        "tensors": len(ctx.manifest()["tensors"]),
        "ok": all(check["status"] == "pass" for check in checks),
        "checks": checks,
    }


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
    # Not argparse-required: --graft takes --source/--base/--out instead, so each
    # mode's own missing-argument path goes through parser.error (clean exit 2),
    # never a TypeError from Path(None).
    parser.add_argument("--src-dir", help="source BF16 checkpoint directory")
    parser.add_argument("--dst-dir", help="packed candidate directory")
    parser.add_argument("--report", help="write the JSON report to this path")
    parser.add_argument(
        "--dequant-sample",
        type=int,
        default=DEFAULT_DEQUANT_SAMPLE,
        help="rows re-decoded by the dequant check (default %(default)s)",
    )
    parser.add_argument(
        "--graft",
        action="store_true",
        help="verify an MTP graft (see tools/rvn_ple/graft_mtp.py); --source, "
             "--base and --out replace --src-dir/--dst-dir",
    )
    parser.add_argument("--source", help="graft mode: MTP source checkpoint")
    parser.add_argument("--base", help="graft mode: ungrafted base candidate")
    parser.add_argument("--out", help="graft mode: grafted checkpoint directory")
    args = parser.parse_args(argv)

    if args.graft:
        wanted = [
            flag
            for flag, value in (
                ("--source", args.source),
                ("--base", args.base),
                ("--out", args.out),
            )
            if not value
        ]
        if wanted:
            parser.error("--graft needs " + ", ".join(wanted))
        if args.src_dir or args.dst_dir:
            parser.error("--graft takes --source/--base/--out, not --src-dir/--dst-dir")
        report = run_graft_verification(args.source, args.base, args.out)
    else:
        if not args.src_dir or not args.dst_dir:
            parser.error(
                "verification needs --src-dir and --dst-dir, or --graft with "
                "--source/--base/--out"
            )
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

#!/usr/bin/env python3
"""Bounded-memory, resumable BF16 -> packed-NVFP4 PLE converter (schema v1).

Conforms to ``docs/rvn-ple-storage-schema.md``. Two passes over the source
checkpoint in <= ``--chunk-mib`` row slices: (1) scan records the table amax
and per-part maxima, then freezes ``g = amax / (6 * 448)`` (neutral 1.0 for an
all-zero table) into the state key BEFORE any quantised write; (2) encodes
each partition into ``rvn_ple_parts/part-NNNNN.safetensors``
(``rvn_ple.packed.w{part}`` / ``rvn_ple.packed.s{part}``) with atomic publish
(``*.tmp`` -> fsync -> rename -> state append). Assemble then hardlinks
unchanged shards, rewrites mixed shards by value, updates the model index and
writes ``ple_storage.json`` LAST.

Resume: ``--resume`` continues only when the whole state key matches
(tampered/stale keys are refused, not silently restarted); without
``--resume`` an existing state is discarded and conversion restarts from
scan (schema §3).

Memory contract (schema §5): only one bounded source/encoded chunk is alive
at a time, no list of all parts is held, part payload bytes stream straight
to the output file, and the source checkpoint is never mutated.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import quant  # noqa: E402

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "ple_storage.json"
PARTS_DIR = "rvn_ple_parts"
REQUIRED_LOADER_FEATURE = "ple-packed-nvfp4-v1"
_DTYPE_SIZE = {"F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1, "U8": 1, "I8": 1,
               "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4,
               "I64": 8, "F64": 8}
_HASH_BLOCK = 1 << 20


def _st_header(path):
    """Parse a safetensors header only. Returns (entries, data_start)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    if len(raw) != n:
        raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw)
    data_start = 8 + n
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    return entries, data_start


def _st_header_blob(entries_ordered):
    """entries_ordered: [(name, dtype, shape, begin, end)] -> padded header bytes."""
    header = {name: {"dtype": dt, "shape": list(shape), "data_offsets": [b, e]}
              for name, dt, shape, b, e in entries_ordered}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    return struct.pack("<Q", len(blob)) + blob


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_write_json(path, obj):
    _atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=False) + "\n").encode())


def _copy_range(src_f, dst_f, begin, nbytes):
    src_f.seek(begin)
    left = nbytes
    while left:
        block = src_f.read(min(left, _HASH_BLOCK))
        if not block:
            raise ValueError("truncated safetensors payload")
        dst_f.write(block)
        left -= len(block)


def _load_spec(path, src_dir):
    spec = json.loads(Path(path).read_text())
    cols = spec.get("cols")
    rows_total = spec.get("logical_rows")
    parts = spec.get("partitioning")
    if not isinstance(cols, int) or cols <= 0 or cols % quant.GROUP_SIZE:
        raise ValueError(f"cols must be a positive multiple of {quant.GROUP_SIZE}: {cols!r}")
    if not isinstance(rows_total, int) or rows_total <= 0:
        raise ValueError(f"logical_rows must be positive: {rows_total!r}")
    if not isinstance(parts, list) or not parts:
        raise ValueError("partitioning must be a non-empty list")
    cursor = 0
    for i, p in enumerate(parts):
        if p.get("part") != i:
            raise ValueError(f"partitioning must be numeric part order starting at 0: entry {i}")
        rows = p.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"part {i}: rows must be positive")
        if p.get("row_offset") != cursor:
            raise ValueError(f"part {i}: row_offset {p.get('row_offset')} != {cursor} "
                             "(partitioning must be a complete non-overlapping cover)")
        cursor += rows
    if cursor != rows_total:
        raise ValueError(f"partitioning covers {cursor} rows, logical_rows is {rows_total}")
    tensor_rows = {}
    for p in parts:
        key = (p["source_shard"], p["source_tensor"])
        if key not in tensor_rows:
            entries, _ = _st_header(Path(src_dir) / p["source_shard"])
            e = entries.get(p["source_tensor"])
            if e is None:
                raise ValueError(f"tensor {p['source_tensor']!r} not found in {p['source_shard']}")
            if e["dtype"] != "BF16" or len(e["shape"]) != 2 or e["shape"][1] != cols:
                raise ValueError(f"tensor {p['source_tensor']!r} must be BF16 [rows, {cols}], "
                                 f"got {e['dtype']} {e['shape']}")
            tensor_rows[key] = e["shape"][0]
    # Each source tensor is tiled by exactly its entries, slice origin at the
    # tensor's minimum row_offset (ratified with WP4): so source slicing is
    # relative to the tensor, never to the logical table offset.
    bases = {}
    for p in parts:
        key = (p["source_shard"], p["source_tensor"])
        bases[key] = min(bases.get(key, p["row_offset"]), p["row_offset"])
    covered = {key: 0 for key in bases}
    for p in sorted(parts, key=lambda q: (q["source_shard"], q["source_tensor"], q["row_offset"])):
        key = (p["source_shard"], p["source_tensor"])
        if p["row_offset"] - bases[key] != covered[key]:
            raise ValueError(
                f"source tensor {key[1]!r} entries must tile it contiguously: "
                f"gap/overlap at row_offset {p['row_offset']}")
        covered[key] += p["rows"]
    for key, rows in covered.items():
        if rows != tensor_rows[key]:
            raise ValueError(
                f"source tensor {key[1]!r} has {tensor_rows[key]} rows but partitioning "
                f"covers {rows}; the whole tensor must be hashed exactly once")
    for p in parts:
        p["_src_offset"] = p["row_offset"] - bases[(p["source_shard"], p["source_tensor"])]
    spec.setdefault("source_repo", "")
    spec.setdefault("source_revision", "")
    spec.setdefault("retained_rewrites", {})
    return spec


def _state_key(spec, g_bits):
    return {"source_revision": spec["source_revision"],
            "encoder_version": quant.ENCODER_VERSION,
            "global_scale_bits": g_bits,
            "group_size": quant.GROUP_SIZE,
            "reconstruction": quant.RECONSTRUCTION}


def _load_state(path):
    try:
        state = json.loads(Path(path).read_text())
        ok = (isinstance(state, dict) and isinstance(state.get("key"), dict)
              and state.get("phase") in ("scan", "encode", "write", "assemble")
              and isinstance(state.get("completed_parts"), list)
              and all(isinstance(x, int) for x in state["completed_parts"])
              and isinstance(state.get("scan"), dict)
              and isinstance(state["scan"].get("amax"), float)
              and isinstance(state["scan"].get("nonfinite_count"), int)
              and isinstance(state["scan"].get("per_part_max"), dict))
    except (ValueError, KeyError, TypeError, OSError):
        ok = False
    if not ok:
        raise ValueError(f"refuse: state file {path} is invalid or tampered "
                         "(rerun without --resume to restart from scan)")
    return state


def _require_key_matches(state, spec, all_part_ids):
    """Validate a resume state key; refuse on any mismatch or inconsistency."""
    key = state["key"]
    base = _state_key(spec, 0)
    for k, v in base.items():
        if k == "global_scale_bits":
            continue
        if key.get(k) != v:
            raise ValueError(f"refuse: resume state key mismatch on {k!r} "
                             f"({key.get(k)!r} != {v!r}); tampered or stale state")
    phase = state["phase"]
    scan = state["scan"]
    per_part = {int(k): float(v) for k, v in scan["per_part_max"].items()}
    if any(i < 0 for i in per_part) or any(i not in all_part_ids for i in per_part):
        raise ValueError("refuse: resume state scan references unknown parts")
    expect_amax = max(per_part.values()) if per_part else 0.0
    if scan["amax"] != expect_amax:
        raise ValueError("refuse: resume state amax is inconsistent with per-part maxima")
    if phase != "scan":
        if set(per_part) != all_part_ids:
            raise ValueError("refuse: resume state past scan lacks complete per-part stats")
        g = quant.compute_global_scale(expect_amax)
        if key.get("global_scale_bits") != quant.global_scale_bits(g):
            raise ValueError("refuse: resume state key global_scale_bits does not match "
                             "the scanned amax (tampered state)")
        quant.global_scale_from_bits(key["global_scale_bits"])
    return per_part


def _rows_per_chunk(cols, chunk_mib):
    return max(1, (chunk_mib * 1024 * 1024) // (cols * 2))


def _read_rows(path, entry, data_start, r0, r1, cols):
    """Read BF16 rows [r0, r1) of a tensor as float32 [r1-r0, cols]."""
    n = r1 - r0
    begin = data_start + entry["data_offsets"][0] + r0 * cols * 2
    nbytes = n * cols * 2
    with open(path, "rb") as f:
        f.seek(begin)
        buf = bytearray(f.read(nbytes))
    if len(buf) != nbytes:
        raise ValueError(f"truncated source payload in {path}")
    return (torch.frombuffer(buf, dtype=torch.uint16)
            .view(torch.bfloat16).reshape(n, cols).to(torch.float32))


def _reject_nonfinite(x, part, row0):
    if bool(torch.isfinite(x).all()):
        return
    bad = (~torch.isfinite(x)).nonzero()[0].tolist()
    raise ValueError(f"non-finite source value at row {row0 + bad[0]} col {bad[1]} "
                     f"of tensor {part['source_tensor']!r} in {part['source_shard']!r}")


def _iter_chunks(rows, rows_per_chunk):
    r0 = 0
    while r0 < rows:
        r1 = min(r0 + rows_per_chunk, rows)
        yield r0, r1
        r0 = r1


def _write_part(spec, part, g, device, chunk_mib, parts_dir):
    """Encode one partition into part-NNNNN.safetensors; atomic publish. Returns digests."""
    cols = spec["cols"]
    rows = part["rows"]
    final = parts_dir / f"part-{part['part']:05d}.safetensors"
    tmp = parts_dir / (final.name + ".tmp")
    tmp.unlink(missing_ok=True)
    w_len, s_len = rows * (cols // 2), rows * (cols // 16)
    w_name = f"rvn_ple.packed.w{part['part']}"
    s_name = f"rvn_ple.packed.s{part['part']}"
    prefix = _st_header_blob([(w_name, "U8", [rows, cols // 2], 0, w_len),
                              (s_name, "F8_E4M3", [rows, cols // 16], w_len, w_len + s_len)])
    shard = Path(spec["_src_dir"]) / part["source_shard"]
    entries, data_start = _st_header(shard)
    entry = entries[part["source_tensor"]]
    hw, hs = hashlib.sha256(), hashlib.sha256()
    with open(tmp, "wb") as out:
        out.write(prefix)
        base = len(prefix)
        for r0, r1 in _iter_chunks(rows, _rows_per_chunk(cols, chunk_mib)):
            x = _read_rows(shard, entry, data_start, part["_src_offset"] + r0,
                           part["_src_offset"] + r1, cols).to(device)
            packed, scales = quant.encode_chunk(x, g)
            wb = packed.cpu().numpy().tobytes()
            sb = scales.view(torch.uint8).cpu().numpy().tobytes()
            hw.update(wb)
            hs.update(sb)
            out.seek(base + r0 * (cols // 2))
            out.write(wb)
            out.seek(base + w_len + r0 * (cols // 16))
            out.write(sb)
            del x, packed, scales, wb, sb
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, final)
    _fsync_dir(parts_dir)
    gc.collect()
    return hw.hexdigest(), hs.hexdigest()


def _hash_part_payloads(path):
    """Stream one part file, returning (sha256_weights, sha256_scales).

    Tensors are hashed in ascending stored-offset order: weight payload then
    scale payload (schema §4: raw contiguous little-endian payload bytes).
    """
    entries, data_start = _st_header(path)
    digests = []
    for name in sorted(entries, key=lambda n: entries[n]["data_offsets"][0]):
        e = entries[name]
        begin, end = e["data_offsets"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            _copy_range(f, _HashSink([h]), data_start + begin, end - begin)
        digests.append(h.hexdigest())
    if len(digests) != 2:
        raise ValueError(f"refuse: part file {path} does not hold exactly two tensors")
    return digests[0], digests[1]


class _HashSink:
    """File-like sink that only hashes (never buffers)."""

    def __init__(self, hashers):
        self._hashers = list(hashers)

    def write(self, b):
        for h in self._hashers:
            h.update(b)


def _source_table_digest(spec, src_dir):
    """sha256 over concat(payload(part_i)), parts ascending (schema §4).

    ``payload(part_i)`` = raw little-endian BF16 bytes of the SOURCE tensor
    rows covered by the partitioning entry with ``part == i`` (schema §2's
    "canonical tensor-payload concat" of the source table; ratified with WP4
    so the verifier, which takes --src-dir, recomputes the identical value).
    """
    master = hashlib.sha256()
    cols = spec["cols"]
    for p in sorted(spec["partitioning"], key=lambda q: q["part"]):
        shard = Path(src_dir) / p["source_shard"]
        entries, data_start = _st_header(shard)
        e = entries[p["source_tensor"]]
        begin = data_start + e["data_offsets"][0] + p["_src_offset"] * cols * 2
        with open(shard, "rb") as f:
            _copy_range(f, _HashSink([master]), begin, p["rows"] * cols * 2)
    return master.hexdigest()


def _rewrite_shard(src, dst, names):
    """Copy named tensors by value into a new safetensors file (atomic publish)."""
    entries, data_start = _st_header(src)
    tmp = dst.with_name(dst.name + ".tmp")
    cursor = 0
    ordered = []
    for name in names:
        e = entries[name]
        n = _dtype_size(e) * _numel(e["shape"])
        ordered.append((name, e["dtype"], e["shape"], cursor, cursor + n))
        cursor += n
    with open(tmp, "wb") as out:
        out.write(_st_header_blob(ordered))
        with open(src, "rb") as f:
            for name, _dt, _sh, b, e2 in ordered:
                _copy_range(f, out, data_start + entries[name]["data_offsets"][0], e2 - b)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)


def _dtype_size(e):
    try:
        return _DTYPE_SIZE[e["dtype"]]
    except KeyError:
        raise ValueError(f"unsupported safetensors dtype {e['dtype']!r}") from None


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _same_shard_content(src, dst):
    """True iff dst is byte-identical to src: same inode, else bounded streaming
    compare (size first, then _HASH_BLOCK chunks)."""
    ss, ds = src.stat(), dst.stat()
    if (ss.st_dev, ss.st_ino) == (ds.st_dev, ds.st_ino):
        return True
    if ss.st_size != ds.st_size:
        return False
    with open(src, "rb") as a, open(dst, "rb") as b:
        while True:
            ca = a.read(_HASH_BLOCK)
            if not ca:
                return True
            if ca != b.read(_HASH_BLOCK):
                return False


def _assemble(spec, state, src_dir, dst_dir, g_bits):
    parts = spec["partitioning"]
    ple_names = {p["source_tensor"] for p in parts}
    packed_names = {f"rvn_ple.packed.w{p['part']}" for p in parts}
    packed_names |= {f"rvn_ple.packed.s{p['part']}" for p in parts}
    plan = {}
    rewritten = set()
    for path in sorted(src_dir.glob("*.safetensors")):
        names, _ = _st_header(path)
        retained = [n for n in names if n not in ple_names]
        if retained:
            plan[path.name] = retained
            if len(retained) != len(names):
                rewritten.add(path.name)  # mixed shard: PLE tensors dropped -> rewritten
    decl = spec["retained_rewrites"]
    retained_all = {n for names in plan.values() for n in names}
    # Only rewritten (mixed) shards require a retained_rewrites declaration;
    # fully-unchanged shards are hardlinked by value below and need none.
    for fname in sorted(rewritten):
        for n in plan[fname]:
            if n not in decl.get(fname, []):
                raise ValueError(
                    f"refuse: retained tensor {n!r} in output shard {fname} is not declared "
                    "in tensors-file retained_rewrites")
    # Guard: the dst tree may only carry (a) shards byte-identical to their source
    # shard (unchanged pass-through hardlinks/copies) or (b) declared retained
    # rewrites. Foreign or tampered files are refused.
    for path in sorted(dst_dir.glob("*.safetensors")):
        names, _ = _st_header(path)
        for n in names:
            if n in ple_names or n in packed_names:
                raise ValueError(
                    f"refuse: dst tensor name collision {n!r} in {path.name} "
                    "collides with a PLE output tensor")
        src = src_dir / path.name
        if src.exists() and _same_shard_content(src, path):
            continue  # unchanged pass-through (hardlink or byte copy)
        if path.name in decl and all(n not in retained_all or n in decl[path.name]
                                    for n in names):
            continue  # declared retained rewrite
        if src.exists():
            raise ValueError(
                f"refuse: dst shard {path.name} is not byte-identical to its source shard "
                "and is not a declared retained rewrite (tampered)")
        for n in names:
            if n in retained_all:
                raise ValueError(
                    f"refuse: dst tensor name collision {n!r} in {path.name} "
                    "is not a declared retained rewrite")
        raise ValueError(
            f"refuse: foreign shard {path.name} in dst has no source counterpart")

    for fname, names in plan.items():
        src = src_dir / fname
        dst = dst_dir / fname
        entries, _ = _st_header(src)
        if dst.exists():
            dst.unlink()
        if set(names) == set(entries):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            _rewrite_shard(src, dst, names)

    weight_map = {}
    total = 0
    for fname, names in plan.items():
        entries, _ = _st_header(src_dir / fname)
        for n in names:
            weight_map[n] = fname
            total += _dtype_size(entries[n]) * _numel(entries[n]["shape"])
    for p in parts:
        i = p["part"]
        f = f"{PARTS_DIR}/part-{i:05d}.safetensors"
        weight_map[f"rvn_ple.packed.w{i}"] = f
        weight_map[f"rvn_ple.packed.s{i}"] = f
        total += p["rows"] * (spec["cols"] // 2) + p["rows"] * (spec["cols"] // quant.GROUP_SIZE)
    _atomic_write_json(dst_dir / INDEX_NAME,
                       {"metadata": {"total_size": total}, "weight_map": weight_map})

    manifest_parts = []
    for p in sorted(parts, key=lambda p: p["part"]):
        f = dst_dir / PARTS_DIR / f"part-{p['part']:05d}.safetensors"
        if not f.exists():
            raise ValueError(f"refuse: packed part {f} is missing")
        dw, ds = _hash_part_payloads(f)
        manifest_parts.append({
            "file": f"{PARTS_DIR}/part-{p['part']:05d}.safetensors",
            "weight_tensor": f"rvn_ple.packed.w{p['part']}",
            "scale_tensor": f"rvn_ple.packed.s{p['part']}",
            "row_offset": p["row_offset"], "rows": p["rows"],
            "sha256_weights": dw, "sha256_scales": ds,
            "first_source_tensor": p["source_tensor"],
            "last_source_tensor": p["source_tensor"],
        })
    manifest = {
        "format_version": 1,
        "encoder_version": quant.ENCODER_VERSION,
        "required_loader_feature": REQUIRED_LOADER_FEATURE,
        "source": {"repo": spec["source_repo"], "revision": spec["source_revision"],
                   "source_table_sha256": _source_table_digest(spec, src_dir),
                   "source_dtype": "bfloat16", "amax": state["scan"]["amax"]},
        "table": {"logical_rows": spec["logical_rows"], "cols": spec["cols"],
                  "partitioning": [{"part": p["part"], "source_shard": p["source_shard"],
                                    "source_tensor": p["source_tensor"],
                                    "row_offset": p["row_offset"], "rows": p["rows"]}
                                   for p in parts]},
        "encoding": {"weight_dtype": quant.WEIGHT_DTYPE, "scale_dtype": quant.SCALE_DTYPE,
                     "group_size": quant.GROUP_SIZE, "scale_layout": "row-major",
                     "global_scale_bits": g_bits, "reconstruction": quant.RECONSTRUCTION},
        "parts": manifest_parts,
        "retained_rewrites": {f: list(names) for f, names in sorted(plan.items())},
    }
    existing = dst_dir / MANIFEST_NAME
    if existing.exists():
        try:
            prior = json.loads(existing.read_text())
        except ValueError:
            prior = None
        if prior != manifest:
            raise ValueError(
                f"refuse: existing {MANIFEST_NAME} in {dst_dir} does not match this "
                "conversion (tampered manifest or stale output tree)")
        return manifest
    _atomic_write_json(existing, manifest)  # written LAST
    return manifest


def convert(*, src_dir, dst_dir, tensors_file, chunk_mib=256, device="cpu",
            resume=False, state=".rvn_convert_state.json"):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if chunk_mib < 1:
        raise ValueError("chunk-mib must be >= 1")
    spec = _load_spec(tensors_file, src_dir)
    spec["_src_dir"] = str(src_dir)
    state_path = Path(state)
    if not state_path.is_absolute():
        state_path = dst_dir / state_path
    dst_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = dst_dir / PARTS_DIR
    parts_dir.mkdir(parents=True, exist_ok=True)

    state_obj = _load_state(state_path) if (resume and state_path.exists()) else None
    all_parts = {p["part"] for p in spec["partitioning"]}
    if state_obj is not None:
        _require_key_matches(state_obj, spec, all_parts)
    else:
        state_obj = {"key": _state_key(spec, 0), "phase": "scan",
                     "completed_parts": [],
                     "scan": {"amax": 0.0, "nonfinite_count": 0, "per_part_max": {}}}
        _atomic_write_json(state_path, state_obj)

    if state_obj["phase"] == "scan":
        for p in spec["partitioning"]:
            if str(p["part"]) in state_obj["scan"]["per_part_max"]:
                continue
            part_amax = 0.0
            shard = src_dir / p["source_shard"]
            entries, data_start = _st_header(shard)
            entry = entries[p["source_tensor"]]
            for r0, r1 in _iter_chunks(p["rows"], _rows_per_chunk(spec["cols"], chunk_mib)):
                x = _read_rows(shard, entry, data_start, p["_src_offset"] + r0,
                               p["_src_offset"] + r1, spec["cols"]).to(device)
                _reject_nonfinite(x, p, p["row_offset"] + r0)
                part_amax = max(part_amax, float(x.abs().max()))
                del x
            state_obj["scan"]["per_part_max"][str(p["part"])] = part_amax
            state_obj["scan"]["amax"] = max(state_obj["scan"]["amax"], part_amax)
            _atomic_write_json(state_path, state_obj)
            gc.collect()
        amax = max(state_obj["scan"]["per_part_max"].values())
        g_bits = quant.global_scale_bits(quant.compute_global_scale(amax))
        state_obj["key"]["global_scale_bits"] = g_bits
        state_obj["phase"] = "write"
        state_obj["completed_parts"] = []
        _atomic_write_json(state_path, state_obj)
    else:
        g_bits = state_obj["key"]["global_scale_bits"]

    g = quant.global_scale_from_bits(g_bits)
    if state_obj["phase"] == "write":
        missing = [i for i in state_obj["completed_parts"]
                   if not (parts_dir / f"part-{i:05d}.safetensors").exists()]
        if missing:
            raise ValueError(f"refuse: state claims completed parts {missing} but files are missing")
        for p in spec["partitioning"]:
            if p["part"] in state_obj["completed_parts"]:
                continue
            _write_part(spec, p, g, device, chunk_mib, parts_dir)
            state_obj["completed_parts"].append(p["part"])
            state_obj["completed_parts"].sort()
            _atomic_write_json(state_path, state_obj)

    state_obj["phase"] = "assemble"
    _atomic_write_json(state_path, state_obj)
    manifest = _assemble(spec, state_obj, src_dir, dst_dir, g_bits)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--tensors-file", required=True,
                    help="JSON: exact PLE table tensor list + partition map")
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--state", default=".rvn_convert_state.json")
    args = ap.parse_args(argv)
    try:
        manifest = convert(src_dir=args.src_dir, dst_dir=args.dst_dir,
                           tensors_file=args.tensors_file, chunk_mib=args.chunk_mib,
                           device=args.device, resume=args.resume, state=args.state)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"converted {manifest['table']['logical_rows']} rows x {manifest['table']['cols']} cols; "
          f"{len(manifest['parts'])} parts; g_bits={manifest['encoding']['global_scale_bits']:#010x}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Graft an MTP draft head from a source checkpoint onto the RVN candidate.

``--speculative-algorithm NEXTN`` builds its draft layer from the TARGET
checkpoint's own ``mtp.*`` tensors. The RVN packed-NVFP4 candidate
(``/models/rvn-qwen38-ple-nvfp4``) carries none -- 296,238 index entries, zero
names containing ``mtp`` -- and its entry class asserts
``rvn_mtp_count(config) == 0`` (models/qwen4_exp.py:3060) while its loader runs
every weight name through ``reject_non_text_weight_name`` (patch 0047), so patch
0057 correctly refuses NEXTN on it.

This tool builds a grafted checkpoint that DOES carry a draft head. Frozen r1
contract (authoritative). The loader side is NOT this tool's job: on the applied
0047..0057 stack ``reject_non_text_weight_name`` still raises on any name
containing ``mtp``, so a grafted checkpoint only serves after the companion
loader patch exempts ``mtp`` names for exactly this stamp. Until then serving it
fails at weight load -- that is the missing loader rule, not a graft defect:

* ``out/`` holds every base file hardlinked unchanged (base shards,
  ``ple_storage.json`` and every ``rvn_ple_parts/`` part), so the graft costs no
  extra bytes for the 76 GiB target and cannot drift from it;
* ``config.json`` = base config with ``mtp_num_hidden_layers: 1`` plus the stamp
  ``rvn_mtp_graft = {"source": ..., "encoder_version": "rvn-mtp-graft-r1",
  "count": 1}`` -- the activation switch, so it is written LAST;
* ``model.safetensors.index.json`` = base weight map plus every grafted
  ``mtp.*`` tensor pointing at the grafted shard;
* ``mtp_graft.json`` = ``{"source": ..., "tensors": {name: {"file": <source
  shard>, "sha256": <payload sha256>, "dtype": ..., "shape": [...]}}}``;
* the grafted shard holds the SOURCE payload bytes verbatim -- no requantise.
* one output shard per source shard that holds ``mtp`` tensors, named
  ``rvn-mtp-graft-NNNNN-of-NNNNN.safetensors`` (the real graft is one file: all
  4,637 source mtp tensors live in ``model-00034-of-00036.safetensors``);
* the base must be the ungrafted RVN text candidate -- zero MTP layers declared,
  no ``rvn_mtp_graft`` stamp, and no index entry whose name contains ``mtp``.
  r1 refuses such a base instead of silently replacing a draft head it did not
  build, so "base entries + grafted entries" is never ambiguous;
* cross-device ``--out`` degrades every hardlink to a byte copy: that is
  announced before the first byte and refused outright by ``--no-copy``.

Byte identity is the whole point: NEXTN verifies every draft token against the
target, so a mismatched draft can only cost speed, never correctness -- but that
guarantee holds only while the draft is the draft its author trained. Requantising
it would silently trade a lossless rejection sampler for a re-quantised one.

Determinism: tensors are laid out in numeric source order (``mtp.layers.2``
before ``mtp.layers.10``, the same key the release gate uses), the header is
written with fixed separators, and JSON is emitted in a fixed key order, so a
rerun over the same inputs yields byte-identical shard, manifest, index and
config.

Only stdlib is used: safetensors headers are parsed and written structurally
(the approach ``convert.py``/``verify.py`` already take), so no tensor is ever
materialised and a 1.5 GiB draft head streams through a 1 MiB buffer.

``graft()`` takes source/base/out explicitly -- no defaults -- so an importing
caller can never write into the shipped paths by accident; ``main()`` alone holds
the contract defaults, and ``--out`` there still has to be a separate directory.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath

TOOL_NAME = "rvn_ple.graft_mtp"
TOOL_VERSION = "1.0.0"

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
GRAFT_MANIFEST_NAME = "mtp_graft.json"

# Frozen r1 graft literals. The loader rule keys the ``mtp`` exemption off the
# encoder_version AND the count, so these are contract text, not tunables.
GRAFT_STAMP_KEY = "rvn_mtp_graft"
GRAFT_ENCODER_VERSION = "rvn-mtp-graft-r1"
GRAFT_COUNT = 1
GRAFT_MTP_NUM_HIDDEN_LAYERS = 1

MTP_PREFIX = "mtp."
# The r1 graft is one draft layer, numbered 0, exactly as NEXTN's draft layer is
# addressed; any other set means the source draft head is not the one the
# contract describes and must not be blessed with count: 1.
MTP_LAYER_RE = re.compile(r"^mtp\.layers\.(\d+)\.")
MTP_SHARD_RE = re.compile(r"^rvn-mtp-graft-\d+-of-\d+\.safetensors$")

RVN_ARCH = "Qwen4ExpForCausalLM"
RVN_MODEL_TYPE = "qwen4_exp_text"

# The real graft, per the frozen contract; every flag stays overridable.
SOURCE_DEFAULT = "/models/qwen38-flash-next"
BASE_DEFAULT = "/models/rvn-qwen38-ple-nvfp4"
OUT_DEFAULT = "/models/rvn-qwen38-ple-nvfp4-mtp"

_HASH_BLOCK = 1 << 20


class GraftError(Exception):
    """A graft-contract input is missing, unreadable or inconsistent."""


def _require(condition, message):
    if not condition:
        raise GraftError(message)


def _reject_duplicate_keys(pairs):
    """JSON hook rejecting duplicate keys (an index must not double-declare)."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise GraftError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _natural_key(name):
    """Numeric-aware sort key: digit runs compare as integers (2 < 10).

    Deliberately identical to ``convert._natural_key``/``verify._natural_key``:
    the graft must lay tensors out in the same numeric source order the release
    gate insists on, or the two tools disagree about what "sorted" means.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", name)
    )


def _st_header(path: Path):
    """Parse one safetensors header only: ``(entries, data_start, metadata)``."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise GraftError(f"unreadable shard (truncated header): {path}")
            (header_len,) = struct.unpack("<Q", head)
            raw = handle.read(header_len)
    except OSError as exc:
        raise GraftError(f"unreadable shard: {path}: {exc}") from exc
    if len(raw) != header_len:
        raise GraftError(f"unreadable shard (truncated header JSON): {path}")
    try:
        meta = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as exc:
        raise GraftError(f"unreadable shard (bad header JSON): {path}: {exc}") from exc
    _require(isinstance(meta, dict), f"unreadable shard (bad header): {path}")
    metadata = meta.get("__metadata__")
    entries = {k: v for k, v in meta.items() if k != "__metadata__"}
    for name, entry in entries.items():
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("dtype"), str)
            and isinstance(entry.get("shape"), list)
            and isinstance(entry.get("data_offsets"), list)
            and len(entry["data_offsets"]) == 2
            and all(
                isinstance(off, int) and not isinstance(off, bool) and off >= 0
                for off in entry["data_offsets"]
            ),
            f"shard {path} has an unreadable tensor entry for {name!r}",
        )
    return entries, 8 + header_len, metadata


def _st_header_blob(entries_ordered, metadata):
    """``[(name, dtype, shape, begin, end)]`` -> framed header bytes.

    ``entries_ordered`` order is preserved verbatim in the header, which is what
    makes the output byte-reproducible; ``__metadata__`` (when the source shard
    carried one) is emitted first, as the safetensors spec example does.
    """
    header = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}
        for name, dtype, shape, begin, end in entries_ordered
    }
    if isinstance(metadata, dict):
        header = {"__metadata__": metadata, **header}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    return struct.pack("<Q", len(blob)) + blob


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(tmp: Path, dst: Path):
    with open(tmp, "ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
    _atomic_replace(tmp, path)


def _atomic_write_json(path: Path, obj):
    _atomic_write_bytes(
        path, (json.dumps(obj, indent=2, sort_keys=False) + "\n").encode()
    )


def _copy_range(src_f, dst_f, begin, nbytes, digest=None):
    """Stream ``[begin, begin+nbytes)`` of ``src_f`` into ``dst_f``, hashing it."""
    src_f.seek(begin)
    left = nbytes
    while left:
        block = src_f.read(min(left, _HASH_BLOCK))
        if not block:
            raise GraftError("truncated safetensors payload")
        if digest is not None:
            digest.update(block)
        dst_f.write(block)
        left -= len(block)


def _sha256_range(path: Path, start, end) -> str:
    """sha256 of ``[start, end)`` of a file, streamed in ``_HASH_BLOCK`` chunks."""
    _require(end >= start, f"bad payload range {start}..{end} in {path}")
    digest = hashlib.sha256()
    remaining = end - start
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining:
            block = handle.read(min(remaining, _HASH_BLOCK))
            if not block:
                raise GraftError(f"{path}: payload ends before byte {end}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, where):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GraftError(f"missing {where}: {path}: {exc}") from exc
    try:
        obj = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as exc:
        raise GraftError(f"{where} is not valid JSON: {path}: {exc}") from exc
    _require(isinstance(obj, dict), f"{where} is not a JSON object: {path}")
    return obj


def _weight_map(index, *, where):
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and weight_map, f"{where} has no weight_map object")
    for name, shard in weight_map.items():
        _require(
            isinstance(shard, str) and shard != "",
            f"{where} maps {name!r} to a non-string shard",
        )
    return weight_map


def _shard_path(root: Path, rel_name):
    pure = PurePosixPath(rel_name)
    _require(
        pure.parts and not pure.is_absolute() and ".." not in pure.parts,
        f"shard name escapes the checkpoint root: {rel_name!r}",
    )
    return root.joinpath(*pure.parts)


def _base_config(base_dir: Path):
    """Fail closed unless ``base_dir`` is the ungrafted RVN text candidate."""
    config = _read_json_object(base_dir / CONFIG_NAME, where=f"{CONFIG_NAME} (base)")
    _require(
        config.get("architectures") == [RVN_ARCH],
        f"base config is not the RVN text candidate: architectures="
        f"{config.get('architectures')!r}, expected [{RVN_ARCH!r}]",
    )
    _require(
        config.get("model_type") == RVN_MODEL_TYPE,
        f"base config is not the RVN text candidate: model_type="
        f"{config.get('model_type')!r}, expected {RVN_MODEL_TYPE!r}",
    )
    # The same sum the loader's rvn_mtp_count() computes; a base that already
    # declares a draft head is not the checkpoint this contract describes.
    count = int(config.get("mtp_num_hidden_layers", 0) or 0) + int(
        config.get("num_nextn_predict_layers", 0) or 0
    )
    _require(
        count == 0,
        f"base already declares {count} MTP layer(s): the r1 graft is defined "
        "for the ungrafted candidate only",
    )
    _require(
        GRAFT_STAMP_KEY not in config,
        f"base already carries a {GRAFT_STAMP_KEY} stamp: refusing to graft onto "
        "an already-grafted checkpoint",
    )
    return config


def _mtp_inventory(src_dir: Path):
    """``(name, source_shard, dtype, shape, start, end)`` in numeric source order.

    ``start``/``end`` are absolute file offsets into the SOURCE shard, and every
    source shard is opened exactly once. The index alone is never trusted: each
    opened shard is cross-checked so an ``mtp.*`` tensor present in the shard but
    absent from the index is a hard error rather than a silently dropped tensor.
    """
    weight_map = _weight_map(
        _read_json_object(src_dir / INDEX_NAME, where=f"{INDEX_NAME} (source)"),
        where=f"{INDEX_NAME} (source)",
    )
    names = sorted((n for n in weight_map if n.startswith(MTP_PREFIX)), key=_natural_key)
    _require(
        bool(names),
        f"source {src_dir} declares no {MTP_PREFIX}* tensors in {INDEX_NAME}: "
        "there is no draft head to graft",
    )
    by_shard = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    rows = []
    for shard in sorted(by_shard, key=_natural_key):
        path = _shard_path(src_dir, shard)
        _require(
            path.is_file(),
            f"{INDEX_NAME} (source) maps mtp tensors to missing shard {shard}",
        )
        entries, data_start, _metadata = _st_header(path)
        undeclared = sorted(
            {n for n in entries if n.startswith(MTP_PREFIX)} - set(by_shard[shard]),
            key=_natural_key,
        )
        _require(
            not undeclared,
            f"shard {shard} holds {MTP_PREFIX}* tensor(s) {undeclared[:3]} that "
            f"{INDEX_NAME} (source) does not map: grafting from this index would "
            "drop them",
        )
        for name in by_shard[shard]:
            entry = entries.get(name)
            _require(
                entry is not None,
                f"{INDEX_NAME} (source) maps {name!r} to {shard}, which lacks it",
            )
            begin, end = entry["data_offsets"]
            _require(end >= begin, f"shard {shard}: {name!r} has decreasing offsets")
            rows.append(
                (
                    name,
                    shard,
                    entry["dtype"],
                    list(entry["shape"]),
                    data_start + begin,
                    data_start + end,
                )
            )
    return rows


def _require_single_draft_head(names):
    """Contract r1: exactly one draft layer, ``mtp.layers.0``."""
    layers = {m.group(1) for name in names if (m := MTP_LAYER_RE.match(name))}
    _require(
        layers == {"0"},
        f"source draft head is not the one-layer head this contract grafts: "
        f"mtp.layers.* ids {sorted(layers) or 'none'}",
    )


def _link_base(base_dir: Path, out_dir: Path, *, no_copy=False):
    """Hardlink every unchanged base file into ``out_dir``.

    Returns ``(linked, copied_bytes)``. Base subdirectories are walked too, not
    just the top level: the candidate's ``ple_storage.json`` names its packed PLE
    payloads by relative path (``rvn_ple_parts/part-*.safetensors``), so a
    top-level-only loop -- all ``convert._copy_metadata_files`` does -- would
    graft a candidate whose manifest points at missing part files. Dotfiles stay
    converter scratch (``.rvn_convert_state.json``), and ``*.tmp`` is this tool's
    own in-flight debris.
    """
    files = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rel = path.relative_to(base_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if len(rel.parts) == 1 and rel.name in (
            CONFIG_NAME,
            INDEX_NAME,
            GRAFT_MANIFEST_NAME,
        ):
            continue
        _require(
            not MTP_SHARD_RE.match(path.name),
            f"base already holds a grafted MTP shard file {path.name}: refusing "
            "to graft over it",
        )
        files.append((path, rel))
    copied = 0
    if files and not _hardlinks_work(out_dir, files[0][0]):
        # Hardlinking is unavailable -- either --out lives on another filesystem,
        # or a base file cannot be linked by this user (ext4's
        # fs.protected_hardlinks denies hardlinking a file you cannot write, and
        # the shipped candidate's ple_storage.json + PLE parts are root-owned).
        # Every link below then degrades to a byte copy, so the operator hears it
        # BEFORE ~76 GiB disappears into a copy that cannot be cancelled cleanly;
        # --no-copy refuses the degradation outright.
        total = sum(path.stat().st_size for path, _rel in files)
        _require(
            not no_copy,
            f"cannot hardlink {base_dir} into {out_dir}: grafting would copy "
            f"{total} bytes. Run as a user that may link the base files, put "
            "--out where linking works, or drop --no-copy to allow the copy",
        )
        print(
            f"hardlinking unavailable: copying {len(files)} files / {total} bytes",
            file=sys.stderr,
            flush=True,
        )
    for path, rel in files:
        copied += _link_file(path, out_dir.joinpath(*rel.parts))
    return len(files), copied


def _hardlinks_work(out_dir: Path, probe_source: Path):
    """True iff a hardlink from the base can actually be created in ``out_dir``.

    One probe answers both ways this fails (cross-device out dir; unowned/
    unwritable base file under fs.protected_hardlinks) instead of letting the
    first denial land mid-run after GiB of work.
    """
    if probe_source.stat().st_dev != out_dir.stat().st_dev:
        return False
    probe = out_dir / ".rvn-hardlink-probe"
    try:
        os.link(probe_source, probe)
    except OSError:
        return False
    probe.unlink()
    return True


def _link_file(src: Path, dst: Path):
    """Publish ``src`` at ``dst`` by hardlink; return bytes copied instead.

    Zero (the normal answer) means the base inode was shared, not duplicated.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(src, tmp)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            # Cross-device --out, which _link_base announces before the first
            # byte: the only case this tool ever copies for.
            # ponytail: upgrade path is an rsync-style reflink helper if
            # cross-device grafts ever matter.
            shutil.copyfile(src, tmp)
            _atomic_replace(tmp, dst)
            return src.stat().st_size
        if exc.errno in (errno.EPERM, errno.EACCES):
            # fs.protected_hardlinks denies linking a file the caller may read
            # but not write, and the shipped candidate's ple_storage.json plus
            # every rvn_ple_parts/part-* are root-owned. Fail closed naming the
            # culprit: copying here would silently spend ~76 GiB on bytes the
            # operator never asked for, one file at a time.
            raise GraftError(
                f"cannot hardlink {src} into {dst.parent}: {exc.strerror}. Run "
                "the graft as a user that may link the base files (the shipped "
                "candidate needs root), or point --out where linking works"
            ) from exc
        raise
    _atomic_replace(tmp, dst)
    return 0


def _graft_shard(src_shard: Path, rows, out_path: Path):
    """Write one grafted shard from one source shard; return per-tensor digests.

    Payload bytes are copied verbatim from the source, so the grafted tensor is
    the trained tensor bit for bit. Every digest is first taken over the bytes as
    read from the SOURCE, then re-taken over the bytes that actually landed in
    ``out_path``: the manifest may only attest to what the grafted shard holds.
    """
    _entries, _data_start, metadata = _st_header(src_shard)
    # (name, dtype, shape, out_begin, out_end, src_begin, src_end)
    plan = []
    cursor = 0
    for name, _shard, dtype, shape, src_begin, src_end in rows:
        span = src_end - src_begin
        plan.append((name, dtype, shape, cursor, cursor + span, src_begin, src_end))
        cursor += span
    tmp = out_path.with_name(out_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    source_digests = {}
    with open(tmp, "wb") as out:
        out.write(
            _st_header_blob(
                [(n, dt, sh, b, e) for n, dt, sh, b, e, _, _ in plan], metadata
            )
        )
        with open(src_shard, "rb") as handle:
            for name, _dt, _sh, _b, _e, src_begin, src_end in plan:
                digest = hashlib.sha256()
                _copy_range(handle, out, src_begin, src_end - src_begin, digest)
                source_digests[name] = digest.hexdigest()
    _atomic_replace(tmp, out_path)
    # The grafted shard is this tool's own bytes, and it is the ONLY thing in
    # out/ a repair path may ever rewrite: base shards are hardlinks to the
    # shipped candidate, so a byte written there would corrupt it in place.
    _require(
        out_path.stat().st_nlink == 1,
        f"grafted shard {out_path.name} is not a private file "
        f"(st_nlink={out_path.stat().st_nlink}): refusing to publish",
    )
    # Re-read what landed on disk: the manifest may only attest to bytes the
    # grafted shard actually holds.
    written, written_start, _ = _st_header(out_path)
    for name, _dt, _sh, begin, end, _sb, _se in plan:
        entry = written.get(name)
        _require(
            entry is not None,
            f"grafted shard {out_path.name} lost tensor {name!r}",
        )
        _require(
            entry["data_offsets"] == [begin, end],
            f"grafted shard {out_path.name} stored {name!r} at "
            f"{entry['data_offsets']}, expected {[begin, end]}",
        )
        landed = _sha256_range(out_path, written_start + begin, written_start + end)
        _require(
            landed == source_digests[name],
            f"grafted payload of {name!r} is not the source payload: "
            f"{source_digests[name][:16]}... -> {landed[:16]}...",
        )
    return source_digests, cursor


def graft(*, source, base, out, no_copy=False):
    """Build the r1 grafted checkpoint; returns an operator summary."""
    src_dir, base_dir, out_dir = Path(source), Path(base), Path(out)
    _require(src_dir.is_dir(), f"missing source checkpoint directory: {src_dir}")
    _require(
        (src_dir / INDEX_NAME).is_file(),
        f"source {src_dir} has no {INDEX_NAME}: cannot enumerate mtp.* tensors",
    )
    _require(base_dir.is_dir(), f"missing base candidate directory: {base_dir}")
    _require(
        (base_dir / INDEX_NAME).is_file(),
        f"base {base_dir} has no {INDEX_NAME}",
    )
    src_root, base_root, out_root = (p.resolve() for p in (src_dir, base_dir, out_dir))
    for other, other_name in ((src_root, "source"), (base_root, "base")):
        _require(
            out_root != other and other not in out_root.parents,
            f"refusing to graft into the {other_name} checkpoint: --out must be a "
            "separate directory",
        )
    if out_dir.exists():
        _require(out_dir.is_dir(), f"--out exists and is not a directory: {out_dir}")
        # A non-empty directory is only re-graftable when it is a previous graft
        # of ours; otherwise --out would clobber an unrelated checkpoint.
        _require(
            not any(out_dir.iterdir()) or (out_dir / GRAFT_MANIFEST_NAME).is_file(),
            f"--out {out_dir} is a non-empty directory without "
            f"{GRAFT_MANIFEST_NAME}: refusing to write into it",
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = _base_config(base_dir)
    base_index = _read_json_object(base_dir / INDEX_NAME, where=f"{INDEX_NAME} (base)")
    base_map = _weight_map(base_index, where=f"{INDEX_NAME} (base)")
    with_mtp = sorted(n for n in base_map if "mtp" in n)
    _require(
        not with_mtp,
        f"base index already carries {len(with_mtp)} tensor name(s) containing "
        f"'mtp' (e.g. {with_mtp[:3]}): the loader rule only exempts mtp names for "
        "a checkpoint whose draft head came from this encoder",
    )

    rows = _mtp_inventory(src_dir)
    _require_single_draft_head([row[0] for row in rows])
    by_shard = {}
    for row in rows:
        by_shard.setdefault(row[1], []).append(row)

    linked, copied = _link_base(base_dir, out_dir, no_copy=no_copy)
    summary = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "source": str(src_root),
        "base": str(base_root),
        "out": str(out_root),
        "encoder_version": GRAFT_ENCODER_VERSION,
        "tensors": len(rows),
        "grafted_bytes": 0,
        "linked_files": linked,
        "copied_bytes": copied,
        "shards": [],
    }

    digests = {}
    grafted_map = {}
    shard_names = [
        f"rvn-mtp-graft-{index + 1:05d}-of-{len(by_shard):05d}.safetensors"
        for index in range(len(by_shard))
    ]
    total_bytes = 0
    for shard_name, (src_shard, shard_rows) in zip(
        shard_names, sorted(by_shard.items(), key=lambda item: _natural_key(item[0]))
    ):
        out_path = out_dir / shard_name
        shard_digests, nbytes = _graft_shard(
            _shard_path(src_dir, src_shard), shard_rows, out_path
        )
        digests.update(shard_digests)
        total_bytes += nbytes
        for name in shard_digests:
            grafted_map[name] = shard_name
        summary["shards"].append(
            {"file": shard_name, "source": src_shard, "tensors": len(shard_rows),
             "bytes": nbytes}
        )
    summary["grafted_bytes"] = total_bytes

    # The frozen r1 manifest shape: one entry per grafted tensor, keyed by name,
    # pointing back at the SOURCE shard and the payload digest it was copied from.
    manifest = {
        "source": str(src_root),
        "tensors": {
            name: {
                "file": source_shard,
                "sha256": digests[name],
                "dtype": dtype,
                "shape": shape,
            }
            for name, source_shard, dtype, shape in (
                (row[0], row[1], row[2], row[3]) for row in rows
            )
        },
    }
    _atomic_write_json(out_dir / GRAFT_MANIFEST_NAME, manifest)

    merged = dict(base_map)
    for name, shard_name in grafted_map.items():
        _require(
            name not in merged,
            f"grafted tensor {name!r} collides with a base index entry",
        )
        merged[name] = shard_name
    metadata = dict(base_index.get("metadata") or {})
    if isinstance(metadata.get("total_size"), int):
        # Keep the HF invariant total_size == sum of tensor payload bytes, which
        # the grafted tensors otherwise silently understate.
        metadata["total_size"] = metadata["total_size"] + total_bytes
    _atomic_write_json(
        out_dir / INDEX_NAME, {"metadata": metadata, "weight_map": merged}
    )

    # The shipped base also carries an inert nested ``mtp`` sub-object declaring
    # num_hidden_layers: 0 (config.json:83-91). The frozen contract stamps only
    # the flat field, and rvn_mtp_count() reads only flat fields, so this tool
    # must NOT silently "repair" the nested object -- but the contradiction is
    # the loader owner's hazard and must not be silent either.
    nested = base_config.get("mtp")
    nested_depth = nested.get("num_hidden_layers") if isinstance(nested, dict) else None
    if nested_depth is not None and nested_depth != GRAFT_MTP_NUM_HIDDEN_LAYERS:
        summary["nested_mtp_num_hidden_layers"] = nested_depth
        print(
            f"WARNING: {CONFIG_NAME} keeps a nested \"mtp\" sub-object with "
            f"num_hidden_layers={nested_depth!r} while this graft stamps "
            f"mtp_num_hidden_layers={GRAFT_MTP_NUM_HIDDEN_LAYERS}; "
            "rvn_mtp_count() reads only the flat fields, but draft plumbing that "
            "resolves depth through the nested object will see 0 layers",
            file=sys.stderr,
            flush=True,
        )

    # The stamp is the activation switch: the loader only exempts mtp names for a
    # config declaring this graft, so it goes last and a half-written graft stays
    # an ungrafted-looking directory that patch 0057 still refuses to serve.
    stamped = dict(
        base_config,
        mtp_num_hidden_layers=GRAFT_MTP_NUM_HIDDEN_LAYERS,
        **{
            GRAFT_STAMP_KEY: {
                "source": str(src_root),
                "encoder_version": GRAFT_ENCODER_VERSION,
                "count": GRAFT_COUNT,
            }
        },
    )
    _atomic_write_json(out_dir / CONFIG_NAME, stamped)

    summary["index_entries"] = len(merged)
    summary["base_index_entries"] = len(base_map)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Graft a source checkpoint's mtp.* draft head onto the RVN "
            "packed-NVFP4 candidate as a separate, byte-identical checkpoint."
        ),
    )
    parser.add_argument(
        "--source", default=SOURCE_DEFAULT,
        help=f"MTP source checkpoint (default %(default)s)",
    )
    parser.add_argument(
        "--base", default=BASE_DEFAULT,
        help=f"ungrafted RVN candidate (default %(default)s)",
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT,
        help=f"grafted output directory (default %(default)s)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="fail closed if --out is on a different filesystem than --base, "
             "instead of paying for a full byte copy of the base shards",
    )
    args = parser.parse_args(argv)
    try:
        summary = graft(source=args.source, base=args.base, out=args.out,
                        no_copy=args.no_copy)
    except GraftError as exc:
        print(f"graft FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"source: {summary['source']} ({summary['tensors']} mtp.* tensors, "
        f"{summary['grafted_bytes']} bytes, {len(summary['shards'])} shard(s))"
    )
    for shard in summary["shards"]:
        print(
            f"  wrote {shard['file']}: {shard['tensors']} tensors, "
            f"{shard['bytes']} bytes from {shard['source']}"
        )
    print(
        f"linked: {summary['linked_files']} base files (hardlinks)"
        + (f", {summary['copied_bytes']} bytes copied cross-device"
           if summary["copied_bytes"] else "")
    )
    print(
        f"wrote {INDEX_NAME}: {summary['base_index_entries']} base + "
        f"{summary['tensors']} grafted = {summary['index_entries']} entries"
    )
    print(
        f"wrote {GRAFT_MANIFEST_NAME} and {CONFIG_NAME} "
        f"(mtp_num_hidden_layers={GRAFT_MTP_NUM_HIDDEN_LAYERS}, "
        f"{GRAFT_STAMP_KEY}={GRAFT_ENCODER_VERSION}, count={GRAFT_COUNT})"
    )
    print(f"graft OK: {summary['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

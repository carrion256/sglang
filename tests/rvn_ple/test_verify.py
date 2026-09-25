"""Synthetic-candidate tests for tools/rvn_ple/verify.py.

Fixtures are tiny torch+safetensors checkpoints written under ``tmp_path``; the
real RVN checkpoint is never touched and ``convert.py`` is never imported. The
expected digests follow docs/rvn-ple-storage-schema.md §1/§4 directly, derived
from the tensors that were saved, not from the verifier's own byte offsets.
"""

import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "rvn_ple" / "verify.py"

MANIFEST_NAME = "ple_storage.json"
INDEX_NAME = "model.safetensors.index.json"
GROUP = 16  # storage contract §1 group_size
COLS = 32
E4M3_MAX = 448.0  # schema §1 global-scale denominator
GLOBAL_SCALE = 0.5
# Schema §1 freezes g = amax / (6 * 448): a conforming fixture must have its
# source rows actually reach this amax, because the verifier recomputes it.
FIXTURE_AMAX = GLOBAL_SCALE * 6 * E4M3_MAX

PLE_NAME = "model.layers.1.ple.ple_embedding.ngram_embedding"


def _ple_tensor(index):
    """One n-gram table shard tensor name, distinct per source shard."""
    return f"{PLE_NAME}.shard_{index}.weight"


RETAINED_NAMES = (
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.norm.weight",
)
DST_SHARD = "model-00001-of-00001.safetensors"
PARTS_DIR = "rvn_ple_parts"

CHECK_ORDER = (
    "manifest_schema",
    "global_scale_bits",
    "partitioning",
    "parts_integrity",
    "source_table_sha256",
    "non_ple_identity",
    "index_refs",
    "partial_candidate",
    "dequant_sample",
)


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32(value):
    """Round to IEEE-754 binary32, as schema §1 freezes the global scale."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _expected_bits(amax):
    """Schema §1 bits: g = amax/(6*448), neutral 1.0 for an all-zero table."""
    return _bits(1.0 if amax == 0 else _f32(amax / (6.0 * E4M3_MAX)))


def _bf16_roundtrip(value):
    """Value after the one BF16 round-to-nearest-even of the §1 decode."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack(
        "<f", struct.pack("<I", (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000)
    )[0]


def _payload_bytes(tensor):
    """Raw contiguous little-endian payload bytes of a tensor (contract §4)."""
    return bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def _payload_sha(tensor):
    return hashlib.sha256(_payload_bytes(tensor)).hexdigest()


def _source_rows(row, cols=COLS, scale=1.0):
    """One deterministic BF16 source row; ``|value| <= 1.5 * scale``.

    ``scale`` is always a power of two times an integer that keeps every value
    exactly representable in BF16, so the fixture's amax is exact, not rounded.
    """
    return torch.tensor(
        [[((row * cols + col) % 13 * 0.25 - 1.5) * scale for col in range(cols)]],
        dtype=torch.bfloat16,
    )


def _packed_weight(row, cols=COLS):
    """E2M1 codes for one row, low nibble first (contract §1)."""

    def code(col):
        return (row * 7 + col * 5 + 3) & 0x0F

    return bytes(code(2 * k) | (code(2 * k + 1) << 4) for k in range(cols // 2))


def _packed_scales(row, cols=COLS):
    """FP8 E4M3 block scales, positive and finite, one per 16 columns."""
    return bytes(0x30 + ((row + j) % 8) for j in range(cols // GROUP))


def _retained_tensors():
    vector = torch.arange(1, 9, dtype=torch.bfloat16)
    return {
        RETAINED_NAMES[0]: vector.reshape(2, 4),
        RETAINED_NAMES[1]: vector.clone(),
        RETAINED_NAMES[2]: vector.flip(0).clone(),
    }


def _save(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous().clone() for name, tensor in tensors.items()},
        str(path),
    )


def _default_plan(count=11):
    """One partition per source shard, one row each, ascending numeric order.

    Shard numbers run 1..11 so the numeric order (2 < 10) differs from the
    lexicographic one, and every partition names its OWN source tensor: the
    verifier can then bind packed bytes to rows by source tensor at all.
    """
    return [
        {
            "part": index,
            "source_shard": f"model-{index + 1:05d}-of-{count:05d}.safetensors",
            "source_tensor": _ple_tensor(index),
            "row_offset": index,
            "rows": 1,
        }
        for index in range(count)
    ]


def build_fixture(root, plan=None, amax=FIXTURE_AMAX, dst_shards=1):
    """Write a conforming source/candidate pair; return ``(src_dir, dst_dir)``.

    ``amax`` is what the source rows actually reach, so the manifest scale is the
    frozen ``amax / (6 * 448)`` of schema §1 and stays recomputable (§4); the
    verifier fails any candidate whose amax is only self-attested. ``dst_shards``
    spreads the retained tensors over that many candidate shards, which is what
    gives the index rules teeth.
    """
    plan = _default_plan() if plan is None else [dict(entry) for entry in plan]
    root = Path(root)
    src, dst = root / "src", root / "dst"

    # Source checkpoint: PLE slice tensors plus the non-PLE tensors to retain.
    scale = amax / 1.5  # the base grid peaks at 1.5; keep every value BF16-exact
    plan_order = sorted(plan, key=lambda entry: entry["row_offset"])
    source_tensors = {}
    for entry in plan_order:
        key = (entry["source_shard"], entry["source_tensor"])
        rows = [
            _source_rows(row, scale=scale)
            for row in range(entry["row_offset"], entry["row_offset"] + entry["rows"])
        ]
        tensor = torch.cat(rows) if len(rows) > 1 else rows[0]
        source_tensors[key] = (
            torch.cat([source_tensors[key], tensor]) if key in source_tensors else tensor
        )
    reached = max(float(tensor.abs().max()) for tensor in source_tensors.values())
    if reached != float(amax):
        raise AssertionError(
            f"fixture cannot reach amax {amax!r}: the source rows hold "
            f"{reached!r}; choose an amax this grid represents exactly in BF16"
        )
    retained = _retained_tensors()
    by_shard = {}
    for (shard, name), tensor in source_tensors.items():
        by_shard.setdefault(shard, {})[name] = tensor
    for index, name in enumerate(sorted(retained)):
        by_shard.setdefault(plan[index % len(plan)]["source_shard"], {})[
            name
        ] = retained[name]
    for shard, tensors in by_shard.items():
        _save(src / shard, tensors)

    # Candidate: the retained tensors over `dst_shards` rewritten shards, then
    # packed bytes per part.
    dst_names = [
        DST_SHARD
        if dst_shards == 1
        else f"model-{index + 1:05d}-of-{dst_shards:05d}.safetensors"
        for index in range(dst_shards)
    ]
    retained_by_dst = {name: {} for name in dst_names}
    for index, name in enumerate(sorted(retained)):
        retained_by_dst[dst_names[index % dst_shards]][name] = retained[name]
    for shard, tensors in retained_by_dst.items():
        if tensors:
            _save(dst / shard, tensors)
    parts_meta = []
    for entry in sorted(plan, key=lambda item: item["part"]):
        part, offset, rows = entry["part"], entry["row_offset"], entry["rows"]
        weight = torch.tensor(
            [list(_packed_weight(offset + row)) for row in range(rows)],
            dtype=torch.uint8,
        ).reshape(rows, COLS // 2)
        scales = (
            torch.tensor(
                [list(_packed_scales(offset + row)) for row in range(rows)],
                dtype=torch.uint8,
            )
            .reshape(rows, COLS // GROUP)
            .view(torch.float8_e4m3fn)
        )
        weight_name, scale_name = f"rvn_ple.packed.w{part}", f"rvn_ple.packed.s{part}"
        file_name = f"{PARTS_DIR}/part-{part:05d}.safetensors"
        _save(dst / file_name, {weight_name: weight, scale_name: scales})
        parts_meta.append(
            {
                "file": file_name,
                "weight_tensor": weight_name,
                "scale_tensor": scale_name,
                "row_offset": offset,
                "rows": rows,
                "sha256_weights": _payload_sha(weight),
                "sha256_scales": _payload_sha(scales.view(torch.uint8)),
                "first_source_tensor": entry["source_tensor"],
                "last_source_tensor": entry["source_tensor"],
            }
        )

    # §4: concat of the source payload slice each partition covers, ascending part.
    bases = {}
    for entry in plan_order:
        bases.setdefault((entry["source_shard"], entry["source_tensor"]), entry["row_offset"])
    digest = hashlib.sha256()
    for entry in sorted(plan, key=lambda item: item["part"]):
        key = (entry["source_shard"], entry["source_tensor"])
        payload = _payload_bytes(source_tensors[key])
        row_bytes = len(payload) // source_tensors[key].shape[0]
        local = entry["row_offset"] - bases[key]
        digest.update(payload[local * row_bytes : (local + entry["rows"]) * row_bytes])

    manifest = {
        "format_version": 1,
        "encoder_version": "rvn-ple-nvfp4-r1",
        "required_loader_feature": "ple-packed-nvfp4-v1",
        "source": {
            "repo": "0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-NVFP4",
            "revision": "0" * 40,
            "source_table_sha256": digest.hexdigest(),
            "source_dtype": "bfloat16",
            "amax": float(amax),
        },
        "table": {
            "logical_rows": sum(entry["rows"] for entry in plan),
            "cols": COLS,
            "partitioning": sorted(plan, key=lambda item: item["part"]),
        },
        "encoding": {
            "weight_dtype": "e2m1-packed-u8-low-first",
            "scale_dtype": "float8_e4m3fn",
            "group_size": GROUP,
            "scale_layout": "row-major",
            "global_scale_bits": _expected_bits(amax),
            "reconstruction": "bf16_direct",
        },
        "parts": parts_meta,
        "retained_rewrites": {
            shard: sorted(names)
            for shard, names in retained_by_dst.items()
            if names
        },
    }
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (dst / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    name: shard
                    for shard, names in sorted(retained_by_dst.items())
                    for name in names
                },
            },
            indent=2,
        )
        + "\n"
    )
    return src, dst


@pytest.fixture(scope="session")
def pristine(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("rvn-ple-pristine"))


@pytest.fixture
def pair(tmp_path, pristine):
    src, dst = tmp_path / "src", tmp_path / "dst"
    shutil.copytree(pristine[0], src)
    shutil.copytree(pristine[1], dst)
    return src, dst


def verify(tmp_path, src, dst, **kwargs):
    """Run the verifier CLI; return ``(completedprocess, parsed report)``."""
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        str(TOOL),
        "--src-dir",
        str(src),
        "--dst-dir",
        str(dst),
        "--report",
        str(report_path),
    ]
    for name, value in kwargs.items():
        command += [f"--{name.replace('_', '-')}", str(value)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert "Traceback" not in proc.stderr, proc.stderr
    return proc, json.loads(report_path.read_text())


def failed_checks(report):
    return {check["name"] for check in report["checks"] if check["status"] != "pass"}


def failure(report, name):
    """The named check's report entry."""
    return next(check for check in report["checks"] if check["name"] == name)


def mutate_manifest(dst, edit):
    path = dst / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_valid_candidate_passes(tmp_path, pair):
    src, dst = pair
    proc, report = verify(tmp_path, src, dst, dequant_sample=4)
    assert proc.returncode == 0, proc.stdout
    assert report["ok"] is True
    assert all(check["status"] == "pass" for check in report["checks"])


def test_report_lists_every_check_in_order(tmp_path, pair):
    src, dst = pair
    proc, report = verify(tmp_path, src, dst)  # default dequant sample
    assert proc.returncode == 0, proc.stdout
    assert [check["name"] for check in report["checks"]] == list(CHECK_ORDER)
    assert report["dequant_sample"] == 8


def test_truncated_part_fails_parts_integrity(tmp_path, pair):
    src, dst = pair
    part = dst / PARTS_DIR / "part-00007.safetensors"
    part.write_bytes(part.read_bytes()[:-5])
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "parts_integrity" in failed_checks(report)


def test_wrong_part_sha_fails_parts_integrity(tmp_path, pair):
    src, dst = pair
    mutate_manifest(dst, lambda m: m["parts"][0].update({"sha256_weights": "f" * 64}))
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "parts_integrity" in failed_checks(report)


def test_overlapping_partitions_fails_partitioning(tmp_path, pair):
    src, dst = pair
    mutate_manifest(
        dst, lambda m: m["table"]["partitioning"][1].update({"row_offset": 0})
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "partitioning" in failed_checks(report)


def test_lexicographic_partition_order_fails_partitioning(tmp_path, pair):
    src, dst = pair
    mutate_manifest(
        dst,
        lambda m: m["table"].update(
            {
                "partitioning": sorted(
                    m["table"]["partitioning"], key=lambda entry: str(entry["part"])
                )
            }
        ),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    failure = next(
        check for check in report["checks"] if check["name"] == "partitioning"
    )
    assert failure["status"] == "fail"
    assert "numeric-ordered" in failure["reason"]


@pytest.mark.parametrize(
    "scale", [-1.0, 0.0, float("inf"), float("nan")], ids=["negative", "zero", "inf", "nan"]
)
def test_bad_global_scale_bits_fails(tmp_path, pair, scale):
    src, dst = pair
    mutate_manifest(
        dst,
        lambda m: m["encoding"].update({"global_scale_bits": _bits(scale)}),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "global_scale_bits" in failed_checks(report)


def test_missing_retained_tensor_fails_identity(tmp_path, pair):
    src, dst = pair
    tensors = load_file(str(dst / DST_SHARD))
    del tensors["model.norm.weight"]
    _save(dst / DST_SHARD, tensors)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "non_ple_identity" in failed_checks(report)


def test_rewritten_retained_payload_fails_identity(tmp_path, pair):
    src, dst = pair
    tensors = load_file(str(dst / DST_SHARD))
    tensors["model.norm.weight"] = (
        tensors["model.norm.weight"] + torch.ones(8, dtype=torch.bfloat16)
    )
    _save(dst / DST_SHARD, tensors)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    failure = next(
        check for check in report["checks"] if check["name"] == "non_ple_identity"
    )
    assert failure["status"] == "fail"
    assert "payload changed" in failure["reason"]


def test_index_pointing_at_nonexistent_tensor_fails(tmp_path, pair):
    src, dst = pair
    path = dst / INDEX_NAME
    index = json.loads(path.read_text())
    index["weight_map"]["model.layers.9.mlp.ghost.weight"] = DST_SHARD
    path.write_text(json.dumps(index, indent=2) + "\n")
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "index_refs" in failed_checks(report)


def test_manifest_without_part_file_fails_partial_candidate(tmp_path, pair):
    src, dst = pair
    (dst / PARTS_DIR / "part-00005.safetensors").unlink()
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    failure = next(
        check for check in report["checks"] if check["name"] == "partial_candidate"
    )
    assert failure["status"] == "fail"
    assert "no BF16 fallback" in failure["reason"]


def test_part_file_without_manifest_fails_partial_candidate(tmp_path, pair):
    src, dst = pair
    (dst / MANIFEST_NAME).unlink()
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    failure = next(
        check for check in report["checks"] if check["name"] == "partial_candidate"
    )
    assert failure["status"] == "fail"
    assert "packed PLE tensors without" in failure["reason"]


def test_shared_source_tensor_slices_are_hashed_once(tmp_path):
    """Two partitions splitting one source tensor must still digest it once."""
    plan = [
        {
            "part": 0,
            "source_shard": "model-00001-of-00001.safetensors",
            "source_tensor": PLE_NAME,
            "row_offset": 0,
            "rows": 1,
        },
        {
            "part": 1,
            "source_shard": "model-00001-of-00001.safetensors",
            "source_tensor": PLE_NAME,
            "row_offset": 1,
            "rows": 1,
        },
    ]
    src, dst = build_fixture(tmp_path / "split", plan=plan)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 0, proc.stdout
    assert report["ok"] is True

    # Dropping the second partition's rows breaks the whole-table-once rule.
    mutate_manifest(
        dst,
        lambda m: m["table"]["partitioning"][1].update({"row_offset": 0, "rows": 1}),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "source_table_sha256" in failed_checks(report) or "partitioning" in failed_checks(report)


def test_inexact_global_scale_exercises_bf16_rounding(tmp_path):
    """A global scale BF16 cannot hold forces real rounding in both decodes."""
    src, dst = build_fixture(tmp_path / "inexact", amax=1.5)
    proc, report = verify(tmp_path, src, dst, dequant_sample=6)
    assert proc.returncode == 0, proc.stdout
    dequant = next(c for c in report["checks"] if c["name"] == "dequant_sample")
    assert dequant["status"] == "pass"
    assert "6 sampled rows" in dequant["reason"]
    # The pin: this candidate's g is not BF16-representable, so the two decode
    # paths above agreed after rounding rather than by construction.
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    scale = struct.unpack(
        "<f", struct.pack("<I", manifest["encoding"]["global_scale_bits"])
    )[0]
    assert _bf16_roundtrip(scale) != scale


def test_unsupported_loader_feature_fails_manifest_schema(tmp_path, pair):
    src, dst = pair
    mutate_manifest(
        dst,
        lambda m: m.update({"required_loader_feature": "ple-packed-nvfp4-v2"}),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "manifest_schema" in failed_checks(report)


def test_unknown_provenance_is_accepted(tmp_path, pair):
    """The converter may not know the source revision; that is not a defect."""
    src, dst = pair
    mutate_manifest(
        dst,
        lambda m: m["source"].update({"repo": "", "revision": ""}),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 0, proc.stdout


# Each test below reproduced a 9/9 PASS plus a loadable-but-numerically-wrong
# candidate before the named check was tightened; they pin the gate at least as
# strict as the loader it gates.


def test_part_source_tensors_must_match_their_row_range(tmp_path, pair):
    """first/last_source_tensor is a claim about rows, not free prose."""
    src, dst = pair
    mutate_manifest(
        dst,
        lambda m: m["parts"][3].update(
            {"first_source_tensor": _ple_tensor(9), "last_source_tensor": _ple_tensor(9)}
        ),
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "not bound to the rows" in failure(report, "parts_integrity")["reason"]


def test_swapped_part_source_tensors_fail(tmp_path, pair):
    """Swapping two parts' source claims leaves every digest valid."""
    src, dst = pair

    def swap(manifest):
        first, second = manifest["parts"][2], manifest["parts"][5]
        for key in ("first_source_tensor", "last_source_tensor"):
            first[key], second[key] = second[key], first[key]

    mutate_manifest(dst, swap)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "parts_integrity" in failed_checks(report)


def test_part_file_named_twice_fails_parts_integrity(tmp_path, pair):
    src, dst = pair
    mutate_manifest(
        dst, lambda m: m["parts"][1].update({"file": m["parts"][0]["file"]})
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "parts_integrity" in failed_checks(report)


def test_duplicate_part_entry_fails_parts_integrity(tmp_path, pair):
    """Two entries for one part file would make the loader write rows twice."""
    src, dst = pair
    mutate_manifest(dst, lambda m: m["parts"].append(dict(m["parts"][0])))
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "repeats" in failure(report, "parts_integrity")["reason"]


def test_extra_tensor_in_part_fails_parts_integrity(tmp_path, pair):
    """The loader requires exactly the declared pair, so the gate may not allow more."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00004.safetensors"
    tensors = load_file(str(part))
    tensors["rvn_ple.packed.surprise"] = torch.zeros(1, dtype=torch.uint8)
    _save(part, tensors)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "must hold exactly" in failure(report, "parts_integrity")["reason"]


def test_part_payload_range_must_match_dtype_and_shape(tmp_path, pair):
    """A header may not aim a tensor at an arbitrary slice of the file."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00006.safetensors"
    raw = part.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    header["rvn_ple.packed.w6"]["data_offsets"][1] += 1
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    part.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "data_offsets span" in failure(report, "parts_integrity")["reason"]


@pytest.mark.parametrize("byte", [0x80, 0x7F], ids=["sign-bit", "nan-encoding"])
def test_scale_domain_rejects_bytes_outside_the_sample(tmp_path, pair, byte):
    """One bad scale byte away from every sampled row must still fail the gate."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00005.safetensors"
    tensors = load_file(str(part))
    scales = tensors["rvn_ple.packed.s5"].view(torch.uint8).clone()
    scales[0, 0] = byte
    scales = scales.view(torch.float8_e4m3fn)
    tensors["rvn_ple.packed.s5"] = scales
    _save(part, tensors)
    mutate_manifest(
        dst,
        lambda m: m["parts"][5].update({"sha256_scales": _payload_sha(scales)}),
    )
    # dequant_sample=1 samples row 0 only, i.e. never this part's row.
    proc, report = verify(tmp_path, src, dst, dequant_sample=1)
    assert proc.returncode == 1
    assert f"scale byte 0x{byte:02X}" in failure(report, "parts_integrity")["reason"]


def test_manifest_naming_a_candidate_tensor_does_not_exempt_it(tmp_path, pair):
    """A leftover copy of a packed name in the candidate stays identity-checked."""
    src, dst = pair
    shard = dst / DST_SHARD
    tensors = load_file(str(shard))
    name = _ple_tensor(4)
    tensors[name] = torch.zeros(1, COLS, dtype=torch.bfloat16)
    _save(shard, tensors)
    index_path = dst / INDEX_NAME
    index = json.loads(index_path.read_text())
    index["weight_map"][name] = DST_SHARD
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "payload changed" in failure(report, "non_ple_identity")["reason"]


def test_index_required_for_multi_shard_candidate(tmp_path):
    src, dst = build_fixture(tmp_path / "multi", dst_shards=2)
    assert (dst / INDEX_NAME).is_file()
    (dst / INDEX_NAME).unlink()
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "model shards" in failure(report, "index_refs")["reason"]


def test_unindexed_candidate_tensor_fails_index_refs(tmp_path):
    src, dst = build_fixture(tmp_path / "unindexed", dst_shards=2)
    path = dst / INDEX_NAME
    index = json.loads(path.read_text())
    dropped = sorted(index["weight_map"])[-1]
    del index["weight_map"][dropped]
    path.write_text(json.dumps(index, indent=2) + "\n")
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert dropped in failure(report, "index_refs")["reason"]


def test_lexicographic_source_order_fails_partitioning(tmp_path, pair):
    """Renumbering parts 0..n-1 cannot hide a lexicographic source order."""
    src, dst = pair

    def relabel(manifest):
        # PLE shard ids carry no leading zeros (shard_2 vs shard_10), so a
        # lexicographic converter orders the table shard_10 before shard_2.
        entries = sorted(
            manifest["table"]["partitioning"], key=lambda entry: str(entry["source_tensor"])
        )
        cursor = 0
        for part, entry in enumerate(entries):
            entry["part"] = part
            entry["row_offset"] = cursor
            cursor += entry["rows"]
        manifest["table"]["partitioning"] = entries

    mutate_manifest(dst, relabel)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "numeric source order" in failure(report, "partitioning")["reason"]


def test_consistent_amax_and_scale_bits_lie_fails_global_scale(tmp_path, pair):
    """Doubling amax and its bits together is caught: amax is recomputed."""
    src, dst = pair

    def inflate(manifest):
        amax = manifest["source"]["amax"] * 2.0
        manifest["source"]["amax"] = amax
        manifest["encoding"]["global_scale_bits"] = _expected_bits(amax)

    mutate_manifest(dst, inflate)
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "self-attested" in failure(report, "global_scale_bits")["reason"]


def test_scale_bits_inconsistent_with_amax_fail_global_scale(tmp_path, pair):
    """The loader's expected-bits rule is enforced by the gate, not at boot."""
    src, dst = pair
    mutate_manifest(
        dst, lambda m: m["encoding"].update({"global_scale_bits": _bits(0.75)})
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "conflicting global scale bits" in failure(report, "global_scale_bits")["reason"]


@pytest.mark.parametrize(
    "amax", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "inf"]
)
def test_bad_amax_fails_manifest_schema(tmp_path, pair, amax):
    src, dst = pair
    mutate_manifest(dst, lambda m: m["source"].update({"amax": amax}))
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "manifest_schema" in failed_checks(report)


@pytest.mark.parametrize(
    "version",
    ["rvn-ple-ncfp4-r1", "rvn-ple-nvfp4-r2", "rvn-ple-nvfp4-r1-dirty"],
    ids=["typo", "bumped", "suffixed"],
)
def test_unsupported_encoder_version_fails_manifest_schema(tmp_path, pair, version):
    src, dst = pair
    mutate_manifest(dst, lambda m: m.update({"encoder_version": version}))
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "encoder_version" in failure(report, "manifest_schema")["reason"]


def test_part_payload_ranges_must_not_overlap(tmp_path, pair):
    """Correct per-tensor extents are not enough: no two tensors share bytes."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00007.safetensors"
    raw = part.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    # Scale keeps its exact 2-byte extent but is aimed inside the weight payload.
    header["rvn_ple.packed.s7"]["data_offsets"] = [14, 16]
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    part.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 1
    assert "overlaps" in failure(report, "parts_integrity")["reason"]


def _rewrite_header(path, header):
    """Replace a safetensors header in place, leaving the payload untouched."""
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])


def test_header_key_order_is_not_offset_order(tmp_path, pair):
    """Writers may list tensors in any order; disjointness pairs by OFFSET."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00009.safetensors"
    raw = part.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    # Same name -> range mapping, keys emitted high-offset-first.
    _rewrite_header(part, dict(reversed(list(header.items()))))
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 0, proc.stdout
    assert failure(report, "parts_integrity")["status"] == "pass"


def test_part_metadata_does_not_break_pair_exactness(tmp_path, pair):
    """The loader's keys() hides __metadata__, and so must the pair check."""
    src, dst = pair
    part = dst / PARTS_DIR / "part-00002.safetensors"
    tensors = load_file(str(part))
    save_file({name: t.contiguous().clone() for name, t in tensors.items()},
              str(part), metadata={"producer": "rvn-convert"})
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 0, proc.stdout
    assert failure(report, "parts_integrity")["status"] == "pass"


def test_sub_byte_store_code_is_checked_not_refused(tmp_path, pair):
    """safetensors packs two F4 values per element, so its extent is not derivable
    from the shape: the anchor must skip that one rule, not abort the gate."""
    src, dst = pair
    name = "model.layers.0.mlp.experts.0.gate_proj_scale"
    tensor = torch.zeros(8, 2, dtype=torch.float4_e2m1fn_x2)
    source_shard = src / "model-00001-of-00011.safetensors"
    tensors = load_file(str(source_shard))
    tensors[name] = tensor
    _save(source_shard, tensors)
    dst_shard = dst / DST_SHARD
    tensors = load_file(str(dst_shard))
    tensors[name] = tensor.clone()
    _save(dst_shard, tensors)
    index_path = dst / INDEX_NAME
    index = json.loads(index_path.read_text())
    index["weight_map"][name] = DST_SHARD
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    mutate_manifest(
        dst, lambda m: m["retained_rewrites"][DST_SHARD].append(name)
    )
    proc, report = verify(tmp_path, src, dst)
    assert proc.returncode == 0, proc.stdout
    assert failure(report, "non_ple_identity")["status"] == "pass"
    assert failure(report, "parts_integrity")["status"] == "pass"

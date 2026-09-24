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
GLOBAL_SCALE = 0.5

PLE_NAME = "model.layers.1.ple.ple_embedding.ngram_embedding"
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


def _payload_bytes(tensor):
    """Raw contiguous little-endian payload bytes of a tensor (contract §4)."""
    return bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def _payload_sha(tensor):
    return hashlib.sha256(_payload_bytes(tensor)).hexdigest()


def _source_rows(row, cols=COLS):
    """One deterministic BF16 source row (exactly representable values)."""
    return torch.tensor(
        [[(row * cols + col) % 13 * 0.25 - 1.5 for col in range(cols)]],
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
    """One partition per source shard, one row each, ascending numeric order."""
    return [
        {
            "part": index,
            "source_shard": f"model-{index + 1:05d}-of-{count:05d}.safetensors",
            "source_tensor": PLE_NAME,
            "row_offset": index,
            "rows": 1,
        }
        for index in range(count)
    ]


def build_fixture(root, plan=None, global_scale=GLOBAL_SCALE):
    """Write a conforming source/candidate pair; return ``(src_dir, dst_dir)``."""
    plan = _default_plan() if plan is None else [dict(entry) for entry in plan]
    root = Path(root)
    src, dst = root / "src", root / "dst"

    # Source checkpoint: PLE slice tensors plus the non-PLE tensors to retain.
    plan_order = sorted(plan, key=lambda entry: entry["row_offset"])
    source_tensors = {}
    for entry in plan_order:
        key = (entry["source_shard"], entry["source_tensor"])
        rows = [_source_rows(row) for row in range(entry["row_offset"], entry["row_offset"] + entry["rows"])]
        tensor = torch.cat(rows) if len(rows) > 1 else rows[0]
        source_tensors[key] = (
            torch.cat([source_tensors[key], tensor]) if key in source_tensors else tensor
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

    # Candidate: retained tensors in one rewritten shard, packed bytes per part.
    _save(dst / DST_SHARD, retained)
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
            "amax": global_scale * 6 * 448,
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
            "global_scale_bits": _bits(global_scale),
            "reconstruction": "bf16_direct",
        },
        "parts": parts_meta,
        "retained_rewrites": {DST_SHARD: sorted(retained)},
    }
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (dst / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: DST_SHARD for name in sorted(retained)},
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
    """A non-dyadic global scale forces real BF16 rounding in both decodes."""
    src, dst = build_fixture(tmp_path / "inexact", global_scale=0.3)
    proc, report = verify(tmp_path, src, dst, dequant_sample=6)
    assert proc.returncode == 0, proc.stdout
    dequant = next(c for c in report["checks"] if c["name"] == "dequant_sample")
    assert dequant["status"] == "pass"
    assert "6 sampled rows" in dequant["reason"]


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

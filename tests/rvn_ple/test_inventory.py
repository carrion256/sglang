"""Synthetic-checkpoint tests for tools/rvn_ple/inventory.py.

Fixtures are tiny torch+safetensors checkpoints written under tmp_path; the
real RVN checkpoint is never touched. Packed-size expectations follow
docs/rvn-ple-storage-schema.md §1 directly, independently of the tool code.
"""

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "rvn_ple" / "inventory.py"
INDEX_NAME = "model.safetensors.index.json"
GROUP = 16  # storage contract §1 group_size

_spec = importlib.util.spec_from_file_location("rvn_ple_inventory", TOOL)
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

SHARD_A = "model-of-2.safetensors"
SHARD_B = "model-of-10.safetensors"

_PLE = "model.layers.1.ple.ple_embedding.ngram_embedding"

# A fixture designed to classify fully: every tensor lands in one of the seven
# real categories and `unmatched` stays empty.
EXPECTED_CATEGORIES = {
    "model.layers.0.self_attn.q_proj.weight": "attention",
    "model.layers.0.mlp.experts.gate_up_proj.weight": "expert_weight",
    "model.layers.0.mlp.experts.gate_up_proj.weight_scale": "expert_scale",
    "model.layers.0.mlp.experts.7.down_proj.weight_scale_inv": "expert_scale",
    "model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_expert",
    "model.layers.0.input_layernorm.weight": "routing_norm",
    "model.layers.0.mlp.gate.weight": "routing_norm",
    "model.layers.0.mlp.gate.e_score_correction_bias": "routing_norm",
    "model.layers.0.attn_hyper_connection.hc_norm.weight": "routing_norm",
    "model.embed_tokens.weight": "embedding_head",
    "lm_head.weight": "embedding_head",
    f"{_PLE}.shard_1.weight": "ple_embedding_table",
    f"{_PLE}.shard_1.weight_scale": "ple_embedding_table",
    f"{_PLE}.shard_2.weight": "ple_embedding_table",
    f"{_PLE}.shard_2.weight_scale": "ple_embedding_table",
    f"{_PLE}.shard_10.weight": "ple_embedding_table",
    f"{_PLE}.shard_10.weight_scale": "ple_embedding_table",
    f"{_PLE}.weight_scale": "ple_embedding_table",
    f"{_PLE}.weight_scale_2": "ple_embedding_table",
    "model.layers.1.ple.key_proj.weight": "ple_other",
    "model.layers.1.ple.conv1d.weight": "ple_other",
}


def _expected_packed_for(name, dtype, shape, byte_size):
    if "scale" in name or dtype == "U8":
        return byte_size
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    cols = shape[-1]
    assert cols % GROUP == 0
    return rows * cols // 2 + rows * (cols // GROUP)


def _tensors():
    return {
        SHARD_A: {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                8, 8, dtype=torch.bfloat16
            ),
            # Fused 4D expert payload + block scales (uint8 nibbles / fp8).
            "model.layers.0.mlp.experts.gate_up_proj.weight": torch.zeros(
                2, 3, 4, 4, dtype=torch.uint8
            ),
            "model.layers.0.mlp.experts.gate_up_proj.weight_scale": torch.zeros(
                2, 1, 4, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.7.down_proj.weight_scale_inv": torch.zeros(
                2, 1, 4, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(
                8, 8, dtype=torch.bfloat16
            ),
            "model.layers.0.input_layernorm.weight": torch.zeros(
                8, dtype=torch.float32
            ),
            "model.layers.0.mlp.gate.weight": torch.zeros(
                4, 8, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(
                4, dtype=torch.float32
            ),
            "model.layers.0.attn_hyper_connection.hc_norm.weight": torch.zeros(
                8, dtype=torch.float32
            ),
            # Unpacked bf16 partition: packing must halve payload + add scales.
            f"{_PLE}.shard_2.weight": torch.zeros(
                32, 64, dtype=torch.bfloat16
            ),
            f"{_PLE}.shard_2.weight_scale": torch.zeros(
                32, 4, dtype=torch.bfloat16
            ),
            # Already-packed uint8 nibble partition + fp8-e4m3 block scales.
            f"{_PLE}.shard_1.weight": torch.zeros(8, 32, dtype=torch.uint8),
            f"{_PLE}.shard_1.weight_scale": torch.zeros(
                8, 2, dtype=torch.float8_e4m3fn
            ),
            "model.embed_tokens.weight": torch.zeros(
                16, 8, dtype=torch.bfloat16
            ),
        },
        SHARD_B: {
            f"{_PLE}.shard_10.weight": torch.zeros(
                16, 64, dtype=torch.bfloat16
            ),
            f"{_PLE}.shard_10.weight_scale": torch.zeros(
                16, 4, dtype=torch.bfloat16
            ),
            f"{_PLE}.weight_scale": torch.zeros(1, dtype=torch.bfloat16),
            f"{_PLE}.weight_scale_2": torch.zeros(1, dtype=torch.float32),
            "model.layers.1.ple.key_proj.weight": torch.zeros(
                8, 8, dtype=torch.bfloat16
            ),
            "model.layers.1.ple.conv1d.weight": torch.zeros(
                8, 1, 4, dtype=torch.float32
            ),
            "lm_head.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
        },
    }


def _write_ckpt(root: Path, shards, weight_map=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "qwen4_exp", "hc_count": 4}\n')
    (root / "tokenizer.json").write_text('{"tokenizer": "tiny"}\n')
    (root / "tokenizer_config.json").write_text('{"model_max_length": 32768}\n')
    (root / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }}{% endfor %}\n"
    )
    (root / "ple_storage.json").write_text('{"format_version": 1}\n')
    for fname, tensors in shards.items():
        save_file(tensors, str(root / fname))
    if weight_map is None:
        weight_map = {
            name: fname for fname, tensors in shards.items() for name in tensors
        }
    (root / INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {"total_size": 0}, "weight_map": weight_map},
            sort_keys=True,
            indent=2,
        )
    )
    return root


@pytest.fixture()
def ckpt(tmp_path):
    return _write_ckpt(tmp_path / "checkpoint", _tensors())


def _run(out, ckpt_dir, extra=()):
    return inv.main(["--dir", str(ckpt_dir), "--out", str(out), *extra])


def _load(path):
    return json.loads(Path(path).read_text())


def test_full_classification_coverage_and_packed_budget(ckpt, tmp_path):
    out = tmp_path / "rvn-inventory.json"
    assert _run(out, ckpt) == 0
    data = _load(out)

    by_name = {t["name"]: t for t in data["tensors"]}
    # Coverage: every indexed tensor emitted exactly once, with a category.
    assert set(by_name) == set(EXPECTED_CATEGORIES)
    assert len(data["tensors"]) == len(EXPECTED_CATEGORIES)
    assert all(t["category"] in inv.CATEGORIES for t in data["tensors"])
    assert {n: t["category"] for n, t in by_name.items()} == EXPECTED_CATEGORIES
    # A fully classifying fixture leaves nothing unmatched.
    assert data["counts"]["by_category"]["unmatched"] == 0
    assert data["counts"]["tensors"] == len(EXPECTED_CATEGORIES)
    assert data["counts"]["shards"] == 2

    # Header-only metadata round-trips (dtypes/shapes/offsets from the header).
    assert by_name[f"{_PLE}.shard_1.weight"]["dtype"] == "U8"
    assert by_name[f"{_PLE}.shard_1.weight_scale"]["dtype"] == "F8_E4M3"
    assert by_name["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"
    assert by_name[f"{_PLE}.shard_2.weight"]["shape"] == [32, 64]
    assert by_name[f"{_PLE}.shard_2.weight"]["shard"] == SHARD_A
    for t in data["tensors"]:
        assert t["byte_size"] == t["data_offsets"][1] - t["data_offsets"][0]

    # Per-category totals agree with the per-tensor entries.
    total = sum(t["byte_size"] for t in data["tensors"])
    budget = data["resource_budget"]
    assert budget["source_total_bytes"] == total
    assert sum(c["bytes"] for c in data["categories"].values()) == total
    for cat in inv.CATEGORIES:
        members = [t for t in data["tensors"] if t["category"] == cat]
        assert data["categories"][cat]["count"] == len(members)
        assert data["categories"][cat]["bytes"] == sum(t["byte_size"] for t in members)
        assert data["categories"][cat]["gb_decimal"] == pytest.approx(
            data["categories"][cat]["bytes"] / 1e9
        )
        assert data["categories"][cat]["gib_binary"] == pytest.approx(
            data["categories"][cat]["bytes"] / 2**30
        )

    # Packed NVFP4 arithmetic per storage contract §1, recomputed here.
    expected_packed = 0
    ple_bytes = 0
    for t in data["tensors"]:
        if t["category"] != "ple_embedding_table":
            assert t["estimated_packed_nvfp4_bytes"] is None
            continue
        want = _expected_packed_for(
            t["name"], t["dtype"], t["shape"], t["byte_size"]
        )
        assert t["estimated_packed_nvfp4_bytes"] == want
        ple_bytes += t["byte_size"]
        expected_packed += want
    # Literal spot checks of the §1 layout: unpacked bf16 32x64 partition and
    # an already-packed uint8 partition.
    assert 32 * 64 // 2 + 32 * (64 // GROUP) == 1152
    assert _expected_packed_for("x.weight", "U8", [8, 32], 256) == 256
    assert budget["ple_embedding_table_bytes"] == ple_bytes
    assert budget["estimated_packed_nvfp4_bytes"] == expected_packed
    assert budget["non_ple_unchanged_bytes"] == total - ple_bytes
    assert budget["candidate_total_bytes"] == (total - ple_bytes) + expected_packed

    # Self-describing output.
    assert data["tool"]["name"] == "rvn_ple.inventory"
    assert data["tool"]["version"] == inv.TOOL_VERSION
    assert set(data["counts"]["by_category"]) == set(inv.CATEGORIES)
    assert "rows*cols/2" in budget["estimate_formula"]


def test_numeric_ordering_not_lexicographic(ckpt, tmp_path):
    out = tmp_path / "rvn-inventory.json"
    assert _run(out, ckpt) == 0
    data = _load(out)

    # Shard files order by the integer of the numeric (of-NNNNN) suffix, never
    # lexicographically (a lexicographic sort puts "of-10" before "of-2").
    assert data["shard_order"] == [SHARD_A, SHARD_B]
    assert sorted([SHARD_A, SHARD_B]) == [SHARD_B, SHARD_A]

    # PLE partitions keep numeric order 1 < 2 < 10 across both name ordering
    # and the recorded partition integers.
    parts = [
        (t["ple_partition"], t["name"])
        for t in data["tensors"]
        if t["ple_partition"] is not None
    ]
    assert [p for p, _ in parts] == [1, 1, 2, 2, 10, 10]
    assert f"{_PLE}.shard_10.weight" < f"{_PLE}.shard_2.weight"  # lexicographic trap


def test_duplicate_tensor_rejected(ckpt, tmp_path):
    dup_dir = tmp_path / "duplicate"
    shards = _tensors()
    # The tensor declared in SHARD_A also appears physically in SHARD_B.
    shards[SHARD_B]["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(
        8, 8, dtype=torch.bfloat16
    )
    _write_ckpt(dup_dir, shards)

    out = tmp_path / "dup.json"
    assert _run(out, dup_dir) != 0
    assert not out.exists()
    with pytest.raises(inv.InventoryError, match="duplicate tensor"):
        inv.build_inventory(dup_dir)


def test_missing_shard_fails(ckpt, tmp_path):
    missing_dir = tmp_path / "missing"
    shards = _tensors()
    weight_map = {
        name: fname for fname, tensors in shards.items() for name in tensors
    }
    weight_map["model.layers.2.self_attn.q_proj.weight"] = "model-of-99.safetensors"
    _write_ckpt(missing_dir, shards, weight_map)

    out = tmp_path / "missing.json"
    assert _run(out, missing_dir) != 0
    assert not out.exists()
    with pytest.raises(inv.InventoryError, match=r"missing shard.*model-of-99"):
        inv.build_inventory(missing_dir)


def test_unpacked_table_with_indivisible_cols_fails(tmp_path):
    bad = tmp_path / "indivisible"
    _write_ckpt(
        bad,
        {
            SHARD_A: {
                f"{_PLE}.shard_2.weight": torch.zeros(8, 20, dtype=torch.bfloat16)
            }
        },
    )
    with pytest.raises(inv.InventoryError, match="not divisible by"):
        inv.build_inventory(bad)


def test_candidate_ple_tensor_names_are_tables():
    # docs/rvn-ple-storage-schema.md §2 candidate tensor names.
    assert inv.classify_tensor("rvn_ple.packed.w0") == ("ple_embedding_table", None)
    assert inv.classify_tensor("rvn_ple.packed.s0") == ("ple_embedding_table", None)


def test_candidate_packed_pair_survives_build_inventory(tmp_path):
    # A real candidate dir (storage contract §2): the e4m3 scale tensor is
    # stored as (rows, dim//16), so its last dim is not divisible by 16 and
    # must NOT be re-packed; both members are already-packed bytes.
    cand = tmp_path / "candidate"
    rows, dim = 16, 128
    _write_ckpt(
        cand,
        {
            SHARD_A: {
                "rvn_ple.packed.w0": torch.zeros(rows, dim // 2, dtype=torch.uint8),
                "rvn_ple.packed.s0": torch.zeros(
                    rows, dim // 16, dtype=torch.float8_e4m3fn
                ),
            }
        },
    )
    data = inv.build_inventory(cand)
    by_name = {t["name"]: t for t in data["tensors"]}
    assert {t["category"] for t in data["tensors"]} == {"ple_embedding_table"}
    assert by_name["rvn_ple.packed.w0"]["estimated_packed_nvfp4_bytes"] == 1024
    assert by_name["rvn_ple.packed.s0"]["estimated_packed_nvfp4_bytes"] == 128
    budget = data["resource_budget"]
    assert budget["estimated_packed_nvfp4_bytes"] == 1024 + 128
    assert budget["candidate_total_bytes"] == 1024 + 128


def test_hashes_deterministic_and_cli_subprocess(ckpt, tmp_path):
    out1 = tmp_path / "inv1.json"
    out2 = tmp_path / "inv2.json"
    # Real CLI entry point, run twice: byte-identical, canonically sorted keys.
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--dir", str(ckpt), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
    assert out1.read_bytes() == out2.read_bytes()
    text = out1.read_text()
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"

    data = _load(out1)
    # Small files (config/tokenizer/chat template/RVN manifest) are always
    # hashed, the index sha256 is recorded, and shard payloads are NOT hashed
    # by default.
    assert data["hashes"]["shard_payloads_hashed"] is False
    assert data["hashes"]["shard_payloads"] == {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "ple_storage.json",
    ):
        expected = hashlib.sha256((ckpt / name).read_bytes()).hexdigest()
        assert data["hashes"]["small_files"][name]["sha256"] == expected
    assert data["index"]["sha256"] == hashlib.sha256(
        (ckpt / INDEX_NAME).read_bytes()
    ).hexdigest()

    out3 = tmp_path / "inv3.json"
    assert _run(out3, ckpt, extra=("--hash-files",)) == 0
    hashed = _load(out3)
    assert hashed["hashes"]["shard_payloads_hashed"] is True
    assert set(hashed["hashes"]["shard_payloads"]) == {SHARD_A, SHARD_B}
    for shard in (SHARD_A, SHARD_B):
        assert hashed["hashes"]["shard_payloads"][shard] == hashlib.sha256(
            (ckpt / shard).read_bytes()
        ).hexdigest()


def test_hyper_connection_is_unmatched_but_norm_rule_holds():
    # Documented rule: HyperConnection mix/combine weights have no contract
    # category -> unmatched (reported honestly); norm-bearing segments ->
    # routing_norm, and the anchored router rule keeps gate_proj out of it.
    assert inv.classify_tensor(
        "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    ) == ("unmatched", None)
    assert inv.classify_tensor(
        "model.layers.0.attn_hyper_connection.hc_norm.weight"
    ) == ("routing_norm", None)
    assert inv.classify_tensor("model.layers.0.mlp.gate_proj.weight") != (
        "routing_norm",
        None,
    )


def test_duplicate_tensor_inside_one_shard_header_fails(tmp_path):
    # A safetensors header can physically repeat a key and json would keep
    # only the last one, so the tool must reject it instead of hiding it.
    bad = tmp_path / "badheader"
    bad.mkdir()
    header = (
        '{"dup.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},'
        '"dup.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}'
    ).encode()
    header += b" " * (-len(header) % 8)
    (bad / SHARD_A).write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    (bad / INDEX_NAME).write_text(json.dumps({"weight_map": {"dup.weight": SHARD_A}}))
    (bad / "config.json").write_text("{}\n")
    with pytest.raises(inv.InventoryError, match="duplicate tensor in shard header"):
        inv.build_inventory(bad)


def test_output_is_never_non_finite(ckpt, tmp_path):
    out = tmp_path / "inv.json"
    assert _run(out, ckpt) == 0
    text = out.read_text()
    # Reject any JSON constant that is not a finite number.
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert "NaN" not in text and "Infinity" not in text
    # The overflow/non-finite guard covers absurd byte counts and bad floats.
    assert inv._finite_float(10**400) == "non-finite"
    assert inv._finite_float(float("inf")) == "non-finite"
    assert inv._finite_float(float("nan")) == "non-finite"
    assert inv._finite_float(2**30) == float(2**30)


def test_non_canonical_ple_shard_id_is_rejected(tmp_path):
    # patches/0047 ple_shard_is_canonical: the loader claims only str(int(id))
    # ids and leaves `shard_01` unclaimed, so the sign-off tool must not relabel
    # it as partition 1 and bless a checkpoint the runtime refuses.
    assert inv.classify_tensor(f"{_PLE}.shard_2.weight") == ("ple_embedding_table", 2)
    with pytest.raises(inv.InventoryError, match="not canonical"):
        inv.classify_tensor(f"{_PLE}.shard_02.weight")
    bad = tmp_path / "noncanon"
    _write_ckpt(
        bad,
        {SHARD_A: {f"{_PLE}.shard_01.weight": torch.zeros(8, 32, dtype=torch.bfloat16)}},
    )
    with pytest.raises(inv.InventoryError, match="shard_01"):
        inv.build_inventory(bad)


def test_payload_extent_must_match_dtype_and_shape(tmp_path):
    # The safetensors reader derives payload length from dtype+shape, and the
    # packed-size budget is computed from them too, so a header claiming a range
    # its dtype+shape cannot fill is rejected rather than inventoried.
    bad = tmp_path / "extent"
    _write_ckpt(bad, {SHARD_A: {"w.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}})
    shard = bad / SHARD_A
    raw = shard.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    header["w.weight"]["data_offsets"][1] += 2
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])
    with pytest.raises(inv.InventoryError, match="data_offsets span"):
        inv.read_shard_header(shard)
    with pytest.raises(inv.InventoryError, match="data_offsets span"):
        inv.build_inventory(bad)


def test_payload_ranges_must_not_overlap(tmp_path):
    """Two tensors claiming the same bytes is not a checkpoint either."""
    bad = tmp_path / "overlap"
    _write_ckpt(bad, {SHARD_A: {"a.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
                                "b.weight": torch.zeros(8, 4, dtype=torch.bfloat16)}})
    shard = bad / SHARD_A
    raw = shard.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    # b.weight keeps its exact 64-byte extent but is aimed at a.weight's bytes.
    header["b.weight"]["data_offsets"] = list(header["a.weight"]["data_offsets"])
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])
    with pytest.raises(inv.InventoryError, match="overlaps"):
        inv.read_shard_header(shard)
    with pytest.raises(inv.InventoryError, match="overlaps"):
        inv.build_inventory(bad)

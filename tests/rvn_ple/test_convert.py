"""Synthetic-checkpoint tests for tools/rvn_ple/convert.py (WP2).

Fixtures are tiny torch+safetensors checkpoints written under tmp_path; the
real RVN checkpoint is never touched. Encoding/manifest expectations follow
docs/rvn-ple-storage-schema.md §1/§2/§4 directly (schema literals re-declared
here), independently of the tool's own helpers.
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
from safetensors.torch import save_file, load_file

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "rvn_ple"

_spec = importlib.util.spec_from_file_location("rvn_ple_convert", TOOL / "convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)
quant = convert.quant  # the one module object the converter actually uses

GROUP = 16
LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # schema §1 magnitude table
PLE = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
G_BITS_ONE = 0x3F800000  # IEEE-754 bits of 1.0f (neutral scale)


def _grid(rows, cols, seed=0):
    """E2M1-grid table: every 16-col block contains 6.0, so block amax == 6.0."""
    gen = torch.Generator().manual_seed(seed)
    lv = torch.tensor(LEVELS)[torch.randint(0, 8, (rows, cols), generator=gen)]
    lv[:, GROUP - 1::GROUP] = 6.0
    return lv.to(torch.bfloat16)


def _shard(directory, name, tensors):
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in tensors.items()},
              str(directory / name))


def _spec_file(tmp, partitioning, cols, logical_rows, name="tensors.json", **extra):
    spec = {"source_repo": "0bserverx/test-rvn", "source_revision": "rev-abc",
            "logical_rows": logical_rows, "cols": cols,
            "partitioning": partitioning}
    spec.update(extra)
    path = tmp / name
    path.write_text(json.dumps(spec))
    return path


def _part(i, shard, tensor, row_offset, rows):
    return {"part": i, "source_shard": shard, "source_tensor": tensor,
            "row_offset": row_offset, "rows": rows}


def _run(src, dst, tensors_file, **kw):
    return convert.convert(src_dir=src, dst_dir=dst, tensors_file=tensors_file, **kw)


def _raw_payloads(path):
    """name -> (dtype, shape tuple, raw payload bytes), parsed from file bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            b, e = v["data_offsets"]
            f.seek(base + b)
            out[k] = (v["dtype"], tuple(v["shape"]), f.read(e - b))
    return out


def _tree_digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- primitives

def test_e2m1_lut_all_16_codes_and_signed_zero():
    for nib in range(16):
        value = quant.e2m1_decode(nib)
        magnitude = LEVELS[nib & 7]
        expected = -magnitude if nib & 8 else magnitude
        assert value == expected or expected == 0.0
        assert quant.e2m1_encode(value) == nib  # signs, incl. -0.0, round-trip
    assert struct.pack("<f", quant.e2m1_decode(0x8)).hex() == "00000080"  # -0.0
    assert struct.pack("<f", quant.e2m1_decode(0x0)).hex() == "00000000"
    assert quant.e2m1_encode(-0.0) == 0x8 and quant.e2m1_encode(0.0) == 0x0
    with pytest.raises(ValueError):
        quant.e2m1_decode(16)


def test_e4m3_subnormal_and_limit_scale_values():
    vals = torch.tensor([[2 ** -9, 448.0, 3 * 2 ** -10, 2 ** -10, 500.0]],
                        dtype=torch.float32)
    back = quant.e4m3_to_f32(quant.e4m3_encode(vals))[0]
    assert float(back[0]) == 2 ** -9          # min subnormal round-trips
    assert float(back[1]) == 448.0            # finite limit
    assert float(back[2]) == 2 ** -8          # RNE between subnormals
    assert float(back[3]) == 0.0              # tie between 0 and min subnormal
    assert float(back[4]) == 448.0            # overflow saturates (no inf/nan)


def test_global_scale_convention_multiply_not_divide():
    g = quant.compute_global_scale(6.0)
    assert g == struct.unpack("<f", struct.pack("<f", 6.0 / (6.0 * 448.0)))[0]
    assert quant.compute_global_scale(0.0) == 1.0
    # One full 16-column block: every nibble code 7 (magnitude 6.0), scale 448.
    packed = torch.full((1, 8), 0x77, dtype=torch.uint8)
    scales = quant.e4m3_encode(torch.tensor([[448.0]], dtype=torch.float32))
    out = quant.reconstruct(packed, scales, g)
    assert torch.equal(out, torch.full((1, 16), 6.0, dtype=torch.bfloat16))
    # The reciprocal convention (code * scale / g) must NOT reproduce the value.
    wrong = quant.reconstruct(packed, scales, 1.0 / g)
    assert not torch.equal(wrong, out)
    assert quant.global_scale_bits(1.0) == G_BITS_ONE
    with pytest.raises(ValueError):
        quant.global_scale_from_bits(quant.global_scale_bits(-1.0))


def test_nibble_low_first_high_second():
    nib = torch.tensor([[1, 7, 0, 8]], dtype=torch.uint8)
    packed = quant.pack_nibbles(nib)
    assert packed.tolist() == [[(7 << 4) | 1, (8 << 4) | 0]]
    assert torch.equal(quant.unpack_nibbles(packed), nib)


# ----------------------------------------------------------------- converter

@pytest.fixture()
def small_src(tmp_path):
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {PLE: _grid(64, 32, seed=7)})
    return src


def test_partition_boundary_rows_chunk_not_dividing(tmp_path):
    rows, cols = 100000, 16  # 1 MiB chunk = 32768 rows: no part/chunk alignment
    src = tmp_path / "src"
    table = _grid(rows, cols, seed=11)
    _shard(src, "model-00001.safetensors", {PLE: table})
    map_a = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 40000),
                                  _part(1, "model-00001.safetensors", PLE, 40000, 30000),
                                  _part(2, "model-00001.safetensors", PLE, 70000, 30000)],
                       cols, rows, name="map_a.json")
    map_b = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, rows)],
                       cols, rows, name="map_b.json")

    dst_a, dst_b = tmp_path / "out_a", tmp_path / "out_b"
    man_a = _run(src, dst_a, map_a, chunk_mib=1)
    man_b = _run(src, dst_b, map_b, chunk_mib=1)

    assert [p["rows"] for p in man_a["table"]["partitioning"]] == [40000, 30000, 30000]
    assert sum(p["rows"] for p in man_a["parts"]) == rows
    assert man_a["encoding"]["global_scale_bits"] == man_b["encoding"]["global_scale_bits"]

    def payloads(dst, man, tensor_key):
        return torch.cat([load_file(str(dst / p["file"]))[p[tensor_key]].view(torch.uint8)
                          for p in man["parts"]])

    wa = payloads(dst_a, man_a, "weight_tensor")
    wb = payloads(dst_b, man_b, "weight_tensor")
    sa = payloads(dst_a, man_a, "scale_tensor")
    sb = payloads(dst_b, man_b, "scale_tensor")
    assert torch.equal(wa, wb) and torch.equal(sa, sb)  # chunking cannot shift bytes
    g = quant.global_scale_from_bits(man_a["encoding"]["global_scale_bits"])
    only = man_b["parts"][0]
    part_b = load_file(str(dst_b / only["file"]))
    recon = quant.reconstruct(part_b[only["weight_tensor"]], part_b[only["scale_tensor"]], g)
    assert torch.equal(recon, table)  # grid values are lossless on the pinned convention


def test_all_zero_table_neutral_scale(tmp_path):
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {PLE: torch.zeros(32, 16, dtype=torch.bfloat16)})
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 32)], 16, 32)
    dst = tmp_path / "out"
    man = _run(src, dst, spec)
    assert man["source"]["amax"] == 0.0
    assert man["encoding"]["global_scale_bits"] == G_BITS_ONE
    payloads = _raw_payloads(dst / man["parts"][0]["file"])
    assert payloads[man["parts"][0]["weight_tensor"]][2] == b"\x00" * (32 * 8)
    assert payloads[man["parts"][0]["scale_tensor"]][2] == b"\x00" * 32


def test_resume_byte_identical_after_interrupt(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {PLE: _grid(128, 16, seed=3)})
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 64),
                                 _part(1, "model-00001.safetensors", PLE, 64, 64)], 16, 128)

    real = quant.encode_chunk
    calls = {"n": 0, "arm": True}

    def flaky(x, g):
        calls["n"] += 1
        if calls["arm"] and calls["n"] == 2:
            raise RuntimeError("simulated mid-run kill")
        return real(x, g)

    monkeypatch.setattr(quant, "encode_chunk", flaky)
    crash_dst = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated"):
        _run(src, crash_dst, spec)
    state_path = crash_dst / ".rvn_convert_state.json"
    state = json.loads(state_path.read_text())
    assert state["phase"] == "write" and state["completed_parts"] == [0]

    calls2 = {"n": 0}

    def counting(x, g):
        calls2["n"] += 1
        return real(x, g)

    monkeypatch.setattr(quant, "encode_chunk", counting)
    _run(src, crash_dst, spec, resume=True)
    assert calls2["n"] == 1  # only the unfinished part was re-encoded

    fresh_dst = tmp_path / "fresh"
    calls["arm"] = False
    _run(src, fresh_dst, spec)
    assert _tree_digests(crash_dst) == _tree_digests(fresh_dst)


def test_tampered_state_key_refuses(tmp_path, small_src):
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 64)], 32, 64)
    dst = tmp_path / "out"
    _run(src=small_src, dst=dst, tensors_file=spec)
    state_path = dst / ".rvn_convert_state.json"
    state = json.loads(state_path.read_text())
    state["key"]["global_scale_bits"] ^= 1
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="global_scale_bits"):
        _run(src=small_src, dst=dst, tensors_file=spec, resume=True)
    state = json.loads(state_path.read_text())
    state["key"]["encoder_version"] = "rvn-ple-nvfp4-r9"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="encoder_version"):
        _run(src=small_src, dst=dst, tensors_file=spec, resume=True)


def test_tampered_manifest_refuses(tmp_path, small_src):
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 64)], 32, 64)
    dst = tmp_path / "out"
    _run(src=small_src, dst=dst, tensors_file=spec)
    manifest_path = dst / "ple_storage.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["parts"][0]["sha256_weights"] = "00" * 32
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="tampered manifest"):
        _run(src=small_src, dst=dst, tensors_file=spec, resume=True)


def test_missing_completed_part_refuses(tmp_path, small_src):
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 32),
                                 _part(1, "model-00001.safetensors", PLE, 32, 32)], 32, 64)
    dst = tmp_path / "out"
    _run(src=small_src, dst=dst, tensors_file=spec)
    victim = dst / "rvn_ple_parts" / "part-00001.safetensors"
    victim.unlink()
    with pytest.raises(ValueError, match="missing"):
        _run(src=small_src, dst=dst, tensors_file=spec, resume=True)


def test_nonfinite_source_rejected_with_row_context(tmp_path):
    src = tmp_path / "src"
    bad = _grid(64, 16, seed=3).clone()
    bad[3, 5] = float("nan")
    _shard(src, "model-00001.safetensors", {PLE: bad})
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 64)], 16, 64)
    with pytest.raises(ValueError, match=r"row 3 col 5"):
        _run(src, tmp_path / "out", spec)


def test_multi_tensor_partition_slices_are_tensor_relative(tmp_path):
    """A partition's source slice is relative to ITS tensor, not the table offset."""
    a, b = PLE + ".a", PLE + ".b"
    ta, tb = _grid(32, 16, seed=2), _grid(32, 16, seed=3)
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {a: ta})
    _shard(src, "model-00002.safetensors", {b: tb})
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", a, 0, 32),
                                 _part(1, "model-00002.safetensors", b, 32, 32)], 16, 64)
    dst = tmp_path / "out"
    man = _run(src, dst, spec)
    g = quant.global_scale_from_bits(man["encoding"]["global_scale_bits"])
    for p, table in zip(man["parts"], (ta, tb)):
        t = load_file(str(dst / p["file"]))
        assert torch.equal(quant.reconstruct(t[p["weight_tensor"]], t[p["scale_tensor"]], g),
                           table)
    h = hashlib.sha256()
    for table in (ta, tb):  # each whole tensor hashed exactly once, parts ascending
        h.update(table.contiguous().view(torch.uint8).numpy().tobytes())
    assert man["source"]["source_table_sha256"] == h.hexdigest()


def test_partial_source_tensor_coverage_refuses(tmp_path):
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {PLE: _grid(64, 16, seed=4)})
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 16)], 16, 16)
    with pytest.raises(ValueError, match="hashed exactly once"):
        _run(src, tmp_path / "out", spec)


# ------------------------------------------------------------------ assemble

@pytest.fixture()
def mixed_src(tmp_path):
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {
        PLE: _grid(64, 16, seed=5),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 8),
    })
    _shard(src, "model-00002.safetensors", {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(8),
    })
    return src


def _mixed_spec(tmp, **extra):
    decl = {"model-00001.safetensors": ["model.layers.0.self_attn.q_proj.weight"],
            "model-00002.safetensors": ["model.layers.0.mlp.gate_proj.weight",
                                        "model.layers.0.input_layernorm.weight"]}
    return _spec_file(tmp, [_part(0, "model-00001.safetensors", PLE, 0, 32),
                            _part(1, "model-00001.safetensors", PLE, 32, 32)],
                      16, 64, retained_rewrites=decl, **extra)


def test_retained_tensors_byte_identity_through_assemble(tmp_path, mixed_src):
    spec = _mixed_spec(tmp_path)
    dst = tmp_path / "out"
    man = _run(mixed_src, dst, spec)

    # unchanged shard: hardlinked, byte-identical
    src2, dst2 = mixed_src / "model-00002.safetensors", dst / "model-00002.safetensors"
    assert dst2.stat().st_ino == src2.stat().st_ino
    assert dst2.read_bytes() == src2.read_bytes()

    # mixed shard: rewritten, retained tensor identity (dtype, shape, payload) preserved
    src_payloads = _raw_payloads(src2)
    dst1_payloads = _raw_payloads(dst / "model-00001.safetensors")
    assert set(dst1_payloads) == {"model.layers.0.self_attn.q_proj.weight"}
    want = _raw_payloads(mixed_src / "model-00001.safetensors")[
        "model.layers.0.self_attn.q_proj.weight"]
    got = dst1_payloads["model.layers.0.self_attn.q_proj.weight"]
    assert (got[0], got[1], hashlib.sha256(got[2]).hexdigest()) == \
           (want[0], want[1], hashlib.sha256(want[2]).hexdigest())

    # Order inside each list follows the source shard header; membership is the contract.
    assert {k: sorted(v) for k, v in man["retained_rewrites"].items()} == {
        "model-00001.safetensors": ["model.layers.0.self_attn.q_proj.weight"],
        "model-00002.safetensors": sorted(["model.layers.0.mlp.gate_proj.weight",
                                           "model.layers.0.input_layernorm.weight"])}
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "rvn_ple.packed.w0", "rvn_ple.packed.s0",
        "rvn_ple.packed.w1", "rvn_ple.packed.s1"}
    assert index["weight_map"]["rvn_ple.packed.w1"] == "rvn_ple_parts/part-00001.safetensors"


def test_undeclared_dst_name_collision_refuses(tmp_path, mixed_src):
    dst = tmp_path / "out"
    _shard(dst, "rogue.safetensors",
           {"model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16)})
    spec = _mixed_spec(tmp_path)
    with pytest.raises(ValueError, match="collision"):
        _run(mixed_src, dst, spec)


@pytest.fixture()
def unchanged_multi_src(tmp_path):
    """Live-scenario shape: one fully-unchanged MULTI-TENSOR shard plus one
    pure-PLE shard, so no mixed shard exists and retained_rewrites stays {}."""
    src = tmp_path / "src"
    _shard(src, "model-00001.safetensors", {
        "lm_head.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    })
    _shard(src, "model-00002.safetensors", {PLE: _grid(32, 16, seed=9)})
    return src


def _unchanged_spec(tmp):
    return _spec_file(tmp, [_part(0, "model-00002.safetensors", PLE, 0, 32)], 16, 32)


def test_unchanged_multi_tensor_shard_hardlinks_with_empty_declarations(tmp_path,
                                                                        unchanged_multi_src):
    """Regression (live failure): retained_rewrites={} is legal when every
    retained shard is fully unchanged and gets hardlinked into dst."""
    spec = _unchanged_spec(tmp_path)  # retained_rewrites stays {}
    dst = tmp_path / "out"
    man = _run(unchanged_multi_src, dst, spec)
    src1 = unchanged_multi_src / "model-00001.safetensors"
    dst1 = dst / "model-00001.safetensors"
    assert (src1.stat().st_dev, src1.stat().st_ino) == \
           (dst1.stat().st_dev, dst1.stat().st_ino)  # same inode: hardlinked
    assert dst1.read_bytes() == src1.read_bytes()
    assert set(_raw_payloads(dst1)) == \
        {"lm_head.weight", "model.layers.0.mlp.gate_proj.weight"}
    assert man["encoding"]["global_scale_bits"] != 0


def test_tampered_hardlinked_shard_refuses(tmp_path, unchanged_multi_src):
    spec = _unchanged_spec(tmp_path)
    dst = tmp_path / "out"
    _run(unchanged_multi_src, dst, spec)
    victim = dst / "model-00001.safetensors"
    data = bytearray(victim.read_bytes())
    victim.unlink()  # copy-up: break the hardlink before mutating anything
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(bytes(data[8:8 + n]))
    off = header["lm_head.weight"]["data_offsets"][0]
    data[8 + n + off] ^= 0xFF  # modify one byte of a retained tensor payload
    victim.write_bytes(data)
    # The source shard must be untouched: the copy-up broke the hardlink.
    assert (unchanged_multi_src / "model-00001.safetensors").read_bytes() != bytes(data)
    with pytest.raises(ValueError, match="not byte-identical"):
        _run(unchanged_multi_src, dst, spec, resume=True)


def test_foreign_dst_shard_refuses(tmp_path, unchanged_multi_src):
    spec = _unchanged_spec(tmp_path)
    dst = tmp_path / "out"
    _shard(dst, "rogue.safetensors",
           {"totally.novel.weight": torch.zeros(2, 2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="foreign"):
        _run(unchanged_multi_src, dst, spec)


def test_source_table_sha256_is_raw_source_payload_concat(tmp_path, mixed_src):
    spec = _mixed_spec(tmp_path)
    dst = tmp_path / "out"
    man = _run(mixed_src, dst, spec)

    # Independent §4 recompute: raw BF16 payload BYTES of each source slice,
    # parsed straight from the shard files, concatenated in ascending part order.
    h = hashlib.sha256()
    for p in sorted(man["table"]["partitioning"], key=lambda q: q["part"]):
        payload = _raw_payloads(mixed_src / p["source_shard"])[p["source_tensor"]][2]
        lo = p["row_offset"] * man["table"]["cols"] * 2
        h.update(payload[lo:lo + p["rows"] * man["table"]["cols"] * 2])
    assert man["source"]["source_table_sha256"] == h.hexdigest()

    # Regression pin: hash-of-hashes (the rejected construction) must differ.
    hh = hashlib.sha256()
    for p in man["parts"]:
        hh.update(bytes.fromhex(p["sha256_weights"]))
        hh.update(bytes.fromhex(p["sha256_scales"]))
    assert man["source"]["source_table_sha256"] != hh.hexdigest()


def test_cli_end_to_end(tmp_path, small_src):
    spec = _spec_file(tmp_path, [_part(0, "model-00001.safetensors", PLE, 0, 64)], 32, 64)
    dst = tmp_path / "cli_out"
    r = subprocess.run(
        [sys.executable, str(TOOL / "convert.py"), "--src-dir", str(small_src),
         "--dst-dir", str(dst), "--tensors-file", str(spec)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (dst / "ple_storage.json").exists()
    assert (dst / "rvn_ple_parts" / "part-00000.safetensors").exists()

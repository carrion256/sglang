"""CPU tests for the manifest-driven packed PLE loader (WP2, patch 0049).

The runtime target files exist only inside
``patches/0049-rvn-ple-packed-loader.patch``; set ``RVN_PLE_TREE`` to the
root of a tree the patch was applied to with ``-p1`` (the throwaway
container does exactly this). Fixtures are synthetic checkpoints under
``tmp_path``; the real RVN checkpoint is never touched. Schema literals
(docs/rvn-ple-storage-schema.md sections 1/2/4) are re-declared here
independently of the loader's own helpers, per the tests/rvn_ple convention.
"""
import hashlib
import importlib.util
import json
import os
import re
import struct
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
_MODULE_REL = Path("python/sglang/srt/models/rvn_ple_storage.py")
_PLE_REL = Path("python/sglang/srt/models/packed_ple.py")
_WU_REL = Path("python/sglang/srt/model_loader/weight_utils.py")

GROUP = 16
LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # schema section 1 magnitudes
E4M3_MAX = 448.0
G_BITS_ONE = 0x3F800000  # IEEE-754 bits of 1.0f (neutral scale)
ROWS_A, ROWS_B = 3, 5
LOGICAL_ROWS = ROWS_A + ROWS_B
COLS = 32
# An awkward float32 (amax-derived) that stresses BF16 rounding behaviour.
G_TIE = 1.166015625
AMAX_TIE = 3134.25  # G_TIE * 6 * 448, exactly representable


def _tree():
    tree = os.environ.get("RVN_PLE_TREE")
    candidates = ([Path(tree)] if tree else []) + [REPO / "runtime"]
    for root in candidates:
        if (root / _MODULE_REL).is_file():
            return root
    raise AssertionError(
        "rvn_ple_storage.py not found: apply patches/0049-rvn-ple-packed-loader"
        ".patch to a tree and set RVN_PLE_TREE to that tree root")


TREE = _tree()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, TREE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rvn = _load("rvn_ple_storage", _MODULE_REL)


# ---------------------------------------------------------------- fixtures


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _e4m3_to_f32(byte):
    """E4M3 (finite, non-negative) -> float, from bit fields only."""
    exp, mant = (byte >> 3) & 0xF, byte & 7
    if exp == 0:
        value = (mant / 8.0) * 2.0 ** -6
    else:
        value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return -value if byte & 0x80 else value


def _bf16_bits_from_f32_bits(bits):
    """BF16 RNE on raw float32 bits (single rounding, schema section 1)."""
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000) >> 16


def _independent_expected_bits(packed, scales, g):
    """Independent dequant per schema literals; returns uint16 BF16 bits.

    Decodes nibbles (low nibble first) and e4m3 scales from raw bytes,
    multiplies through the float32 chain, and rounds to BF16 once.
    """
    rows, half = packed.shape
    cols = half * 2
    scale_bytes = scales.view(torch.uint8)
    out = torch.zeros(rows, cols, dtype=torch.uint16)
    for r in range(rows):
        for c in range(cols):
            byte = int(packed[r, c // 2])
            nib = byte & 0xF if c % 2 == 0 else byte >> 4
            mag = LEVELS[nib & 7]
            val = -mag if nib & 8 else mag
            scale = _e4m3_to_f32(int(scale_bytes[r, c // GROUP]))
            prod = _f32(_f32(val * scale) * g)
            out[r, c] = _bf16_bits_from_f32_bits(_f32_bits(prod))
    return out


def _grid_rows(n, seed=7):
    """Deterministic packed rows covering every E2M1 nibble code."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, COLS), generator=gen, dtype=torch.int64)
    codes[0, :16] = torch.arange(16)
    lo = codes[:, 0::2] & 0xF
    hi = (codes[:, 1::2] & 0xF) << 4
    return (lo | hi).to(torch.uint8)


def _scale_rows(n, seed=11):
    """Deterministic positive-finite e4m3 block scales, no sign/NaN bits."""
    gen = torch.Generator().manual_seed(seed)
    raw = 0x30 + torch.randint(0, 10, (n, COLS // GROUP), generator=gen,
                               dtype=torch.int64)
    return raw.to(torch.uint8).view(torch.float8_e4m3fn)


def _file_payload(path, name):
    """name -> raw little-endian payload bytes, parsed from the file bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        begin, end = header[name]["data_offsets"]
        f.seek(base + begin)
        return f.read(end - begin)


def _write_part(root, index, packed, scales):
    rel = f"rvn_ple_parts/part-{index:05d}.safetensors"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"rvn_ple.packed.w{index}": packed.contiguous().clone(),
               f"rvn_ple.packed.s{index}": scales.contiguous().clone()},
              str(path))
    return {
        "file": rel,
        "weight_tensor": f"rvn_ple.packed.w{index}",
        "scale_tensor": f"rvn_ple.packed.s{index}",
        "row_offset": 0, "rows": int(packed.shape[0]),
        "sha256_weights": hashlib.sha256(_file_payload(path, f"rvn_ple.packed.w{index}")).hexdigest(),
        "sha256_scales": hashlib.sha256(_file_payload(path, f"rvn_ple.packed.s{index}")).hexdigest(),
        "first_source_tensor": "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        "last_source_tensor": "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
    }


def _manifest(parts, amax=AMAX_TIE, g_bits=None, reconstruction="bf16_direct",
              logical_rows=LOGICAL_ROWS, cols=COLS):
    cursor = 0
    partitioning = []
    for index, part in enumerate(parts):
        part["row_offset"] = cursor
        partitioning.append({"part": index,
                             "source_shard": "model-00005-of-00098.safetensors",
                             "source_tensor": "model.layers.1.ple.ple_embedding"
                                              ".ngram_embedding.weight",
                             "row_offset": cursor, "rows": part["rows"]})
        cursor += part["rows"]
    if g_bits is None:
        g_bits = _f32_bits(1.0 if amax == 0.0 else _f32(amax / (6.0 * E4M3_MAX)))
    return {
        "format_version": 1,
        "encoder_version": "rvn-ple-nvfp4-r1",
        "required_loader_feature": "ple-packed-nvfp4-v1",
        "source": {"repo": "0bserverx/test-rvn", "revision": "rev-test",
                   "source_table_sha256": "ab" * 32, "source_dtype": "bfloat16",
                   "amax": amax},
        "table": {"logical_rows": logical_rows, "cols": cols,
                  "partitioning": partitioning},
        "encoding": {"weight_dtype": "e2m1-packed-u8-low-first",
                     "scale_dtype": "float8_e4m3fn", "group_size": GROUP,
                     "scale_layout": "row-major",
                     "global_scale_bits": g_bits,
                     "reconstruction": reconstruction},
        "parts": parts,
        "retained_rewrites": {"model-00001-of-00098.safetensors":
                              ["model.embed_tokens.weight"]},
    }


def _checkpoint(tmp_path, manifest):
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    (root / "ple_storage.json").write_text(json.dumps(manifest))
    return root


def _built_checkpoint(tmp_path):
    """Two valid parts + matching manifest; returns (root, packed, scales)."""
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    packed = torch.cat([_grid_rows(ROWS_A), _grid_rows(ROWS_B, seed=8)])
    scales = torch.cat([_scale_rows(ROWS_A), _scale_rows(ROWS_B, seed=12)])
    parts = [_write_part(root, 0, packed[:ROWS_A], scales[:ROWS_A]),
             _write_part(root, 1, packed[ROWS_A:], scales[ROWS_A:])]
    (root / "ple_storage.json").write_text(json.dumps(_manifest(parts)))
    return root, packed, scales


# ------------------------------------------------------------------- tests


def test_manifest_assembles_packed_host_storage(tmp_path):
    root, packed, scales = _built_checkpoint(tmp_path)
    manifest = rvn.load_manifest(str(root))
    assert manifest.logical_rows == LOGICAL_ROWS and manifest.cols == COLS
    assert manifest.global_scale == _f32(AMAX_TIE / (6.0 * E4M3_MAX))
    storage = rvn.assemble_host_storage(
        manifest, str(root), storage=_load("packed_ple", _PLE_REL)
        .PackedPLEStorage(LOGICAL_ROWS, COLS, pin_memory=False))
    assert torch.equal(storage.weight, packed)
    assert torch.equal(storage.scales.view(torch.uint8),
                       scales.view(torch.uint8))
    assert storage.global_scale == manifest.global_scale
    # Same result through the single-call entry point.
    again = rvn.load_for_checkpoint(
        str(root), storage=_load("packed_ple2", _PLE_REL)
        .PackedPLEStorage(LOGICAL_ROWS, COLS, pin_memory=False))
    assert torch.equal(again.weight, packed)


def test_cpu_gather_reference_parity_single_rounding():
    packed = _grid_rows(4)
    scales = _scale_rows(4)
    out = rvn.dequant_reference(packed, scales, G_TIE)
    expected = _independent_expected_bits(packed, scales, G_TIE)
    _assert_parity(out, expected, "cpu gather reference")


def test_missing_part_file_raises_no_fallback(tmp_path):
    root, _, _ = _built_checkpoint(tmp_path)
    os.remove(root / "rvn_ple_parts" / "part-00001.safetensors")
    with pytest.raises(FileNotFoundError, match="missing"):
        rvn.load_for_checkpoint(
            str(root), storage=_load("packed_ple3", _PLE_REL)
            .PackedPLEStorage(LOGICAL_ROWS, COLS, pin_memory=False))


def test_conflicting_global_scale_bits_raise():
    parts = [{"file": "x.safetensors", "weight_tensor": "w", "scale_tensor": "s",
              "row_offset": 0, "rows": LOGICAL_ROWS,
              "sha256_weights": "0" * 64, "sha256_scales": "0" * 64,
              "first_source_tensor": "a", "last_source_tensor": "b"}]
    # Recorded bits disagree with the frozen g = amax / (6 * 448).
    with pytest.raises(ValueError, match="conflicting global scale"):
        rvn.parse_manifest(_manifest(parts, amax=AMAX_TIE, g_bits=G_BITS_ONE))
    # Bits decoding to a non-positive global scale are rejected outright.
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        rvn.parse_manifest(_manifest(parts, amax=0.0, g_bits=0))
    # The neutral all-zero-table scale is the defined exception, not a clash.
    ok = rvn.parse_manifest(_manifest(parts, amax=0.0, g_bits=G_BITS_ONE))
    assert ok.global_scale == 1.0


def test_row_coverage_hole_raises():
    def part(index, rows):
        return {"file": f"p{index}", "weight_tensor": f"w{index}",
                "scale_tensor": f"s{index}", "row_offset": 0, "rows": rows,
                "sha256_weights": "0" * 64, "sha256_scales": "0" * 64,
                "first_source_tensor": "a", "last_source_tensor": "a"}

    # _manifest() normalizes row offsets, so inject holes after building.
    # Hole in the parts cover: rows 3..4 are never covered.
    hole = _manifest([part(0, ROWS_A), part(1, ROWS_B)])
    hole["parts"][1]["row_offset"] = 5
    with pytest.raises(ValueError, match="row coverage hole"):
        rvn.parse_manifest(hole)
    # Hole in the partitioning cover.
    hole = _manifest([part(0, ROWS_A), part(1, ROWS_B)])
    hole["table"]["partitioning"][0]["row_offset"] = 1
    with pytest.raises(ValueError, match="row coverage hole"):
        rvn.parse_manifest(hole)
    # A cover that never reaches logical_rows is rejected as well.
    with pytest.raises(ValueError, match="logical_rows"):
        rvn.parse_manifest(_manifest([part(0, ROWS_A)]))


def test_absent_manifest_leaves_lil_path_untouched(tmp_path):
    root = tmp_path / "lil"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"untouched")
    assert rvn.load_manifest(str(root)) is None
    assert rvn.load_for_checkpoint(str(root)) is None
    assert (root / "model.safetensors").read_bytes() == b"untouched"


def test_weight_utils_hook_declines_without_manifest(tmp_path):
    sys.path.insert(0, str(TREE / "python"))
    try:
        weight_utils = _load("rvn_wu_probe", _WU_REL)
        assert weight_utils.rvn_ple_storage_for_checkpoint(str(tmp_path)) is None
    finally:
        sys.path.remove(str(TREE / "python"))


def test_swapped_nibble_producer_fails_parity_loudly():
    packed = _grid_rows(4)
    scales = _scale_rows(4)
    expected = _independent_expected_bits(packed, scales, G_TIE)
    _assert_parity(rvn.dequant_reference(packed, scales, G_TIE), expected,
                   "correct producer")
    # A high-first producer swaps the nibble pairs inside every byte.
    swapped = ((packed & 0x0F) << 4) | (packed >> 4)
    with pytest.raises(AssertionError, match="parity"):
        _assert_parity(rvn.dequant_reference(swapped, scales, G_TIE), expected,
                       "swapped-nibble producer")


def test_linear_quant_ple_exclusion_does_not_block_manifest_loader(tmp_path):
    root, packed, scales = _built_checkpoint(tmp_path)
    manifest = json.loads((root / "ple_storage.json").read_text())
    names = [entry["source_tensor"]
             for entry in manifest["table"]["partitioning"]]
    # The checkpoint's generic linear-quant ignore glob does match the PLE
    # names (modelopt glob -> regex semantics, re-declared from the fork).
    glob = "*ple*"
    regex = glob.replace(".", r"\.").replace("*", r".*")
    assert names and all(re.search(regex, name) for name in names)
    storage = rvn.load_for_checkpoint(
        str(root), storage=_load("packed_ple4", _PLE_REL)
        .PackedPLEStorage(LOGICAL_ROWS, COLS, pin_memory=False))
    assert torch.equal(storage.weight, packed)
    assert torch.equal(storage.scales.view(torch.uint8),
                       scales.view(torch.uint8))


def test_reconstruction_modes_stay_selectable(tmp_path):
    parts = [{"file": "x.safetensors", "weight_tensor": "w", "scale_tensor": "s",
              "row_offset": 0, "rows": 1, "sha256_weights": "0" * 64,
              "sha256_scales": "0" * 64, "first_source_tensor": "a",
              "last_source_tensor": "a"}]
    assert rvn.parse_manifest(_manifest(
        parts, logical_rows=1, reconstruction="bf16_direct")).reconstruction \
        == "bf16_direct"
    legacy = rvn.parse_manifest(
        _manifest(parts, logical_rows=1, reconstruction="fp8_roundtrip"))
    assert legacy.reconstruction == "fp8_roundtrip" and legacy.fp8_reference
    with pytest.raises(ValueError, match="reconstruction"):
        rvn.parse_manifest(
            _manifest(parts, logical_rows=1, reconstruction="fp4_roundtrip"))
    # Exact e4m3 products are unaffected by the fp8 round-trip; the reference
    # stays selectable for them.
    packed = torch.full((1, 8), 0x22, dtype=torch.uint8)  # all code 1.0, 16 cols
    scales = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    direct = rvn.dequant_reference(packed, scales, 1.0)
    roundtrip = rvn.dequant_reference(packed, scales, 1.0, fp8_reference=True)
    assert torch.equal(direct.view(torch.uint16), roundtrip.view(torch.uint16))
    assert int(direct.view(torch.uint16)[0, 0]) == _bf16_bits_from_f32_bits(
        _f32_bits(1.0))


def _assert_parity(out, expected_bits, what):
    """Bit-exact BF16 parity; raises AssertionError naming the mismatch."""
    bits = out.contiguous().view(torch.uint16)
    if not torch.equal(bits, expected_bits):
        bad = (bits != expected_bits).nonzero()[0].tolist()
        raise AssertionError(
            f"{what}: parity failed, first mismatch at {bad}: "
            f"got {int(bits[bad[0], bad[1]]):#06x} expected "
            f"{int(expected_bits[bad[0], bad[1]]):#06x}")

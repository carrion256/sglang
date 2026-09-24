"""GPU gather parity for manifest-assembled packed PLE storage (WP2, patch 0049).

Skipped when CUDA is unavailable. DO NOT run while production workers hold
the GPUs: this file launches real Triton kernels. The assembled host bytes
are uploaded explicitly so the check is self-contained on one device.
"""
import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a free CUDA GPU (production workers hold the GPUs)")

REPO = Path(__file__).resolve().parents[2]
_MODULE_REL = Path("python/sglang/srt/models/rvn_ple_storage.py")
_PLE_REL = Path("python/sglang/srt/models/packed_ple.py")

COLS = 32
ROWS = 4
G = 0.3
AMAX = G * 6.0 * 448.0


def _tree():
    tree = os.environ.get("RVN_PLE_TREE")
    if tree:
        root = Path(tree)
        assert (root / _MODULE_REL).is_file(), (
            f"RVN_PLE_TREE={tree} lacks {_MODULE_REL}: apply "
            "patches/0049-rvn-ple-packed-loader.patch to that tree root")
        return root
    if (REPO / "runtime" / _MODULE_REL).is_file():
        return REPO / "runtime"
    pytest.skip(
        "rvn_ple_storage.py ships only inside "
        "patches/0049-rvn-ple-packed-loader.patch; set RVN_PLE_TREE to the "
        "tree the patch was applied to", allow_module_level=True)


TREE = _tree()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, TREE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_payload(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        begin, end = header[name]["data_offsets"]
        f.seek(base + begin)
        return f.read(end - begin)


def _checkpoint(tmp_path):
    import hashlib

    root = tmp_path / "ckpt"
    (root / "rvn_ple_parts").mkdir(parents=True)
    codes = torch.arange(COLS, dtype=torch.int64).remainder(16)
    codes = codes.repeat(ROWS, 1)
    packed = ((codes[:, 1::2] & 0xF) << 4 | (codes[:, 0::2] & 0xF)
              ).to(torch.uint8)
    scale_bytes = torch.full((ROWS, COLS // 16), 0x38, dtype=torch.uint8)
    scales = scale_bytes.view(torch.float8_e4m3fn)
    path = root / "rvn_ple_parts" / "part-00000.safetensors"
    save_file({"rvn_ple.packed.w0": packed.clone(),
               "rvn_ple.packed.s0": scales.clone()}, str(path))
    g_bits = struct.unpack("<I", struct.pack("<f", struct.unpack(
        "<f", struct.pack("<f", AMAX / (6.0 * 448.0)))[0]))[0]
    (root / "ple_storage.json").write_text(json.dumps({
        "format_version": 1,
        "encoder_version": "rvn-ple-nvfp4-r1",
        "required_loader_feature": "ple-packed-nvfp4-v1",
        "source": {"repo": "0bserverx/test-rvn", "revision": "rev-test",
                   "source_table_sha256": "cd" * 32,
                   "source_dtype": "bfloat16", "amax": AMAX},
        "table": {"logical_rows": ROWS, "cols": COLS, "partitioning": [
            {"part": 0, "source_shard": "s.safetensors",
             "source_tensor": "model.layers.1.ple.ple_embedding"
                              ".ngram_embedding.weight",
             "row_offset": 0, "rows": ROWS}]},
        "encoding": {"weight_dtype": "e2m1-packed-u8-low-first",
                     "scale_dtype": "float8_e4m3fn", "group_size": 16,
                     "scale_layout": "row-major",
                     "global_scale_bits": g_bits,
                     "reconstruction": "bf16_direct"},
        "parts": [{"file": "rvn_ple_parts/part-00000.safetensors",
                   "weight_tensor": "rvn_ple.packed.w0",
                   "scale_tensor": "rvn_ple.packed.s0",
                   "row_offset": 0, "rows": ROWS,
                   "sha256_weights": hashlib.sha256(
                       _file_payload(path, "rvn_ple.packed.w0")).hexdigest(),
                   "sha256_scales": hashlib.sha256(
                       _file_payload(path, "rvn_ple.packed.s0")).hexdigest(),
                   "first_source_tensor": "a", "last_source_tensor": "a"}],
        "retained_rewrites": {},
    }))
    return root, packed, scales


def test_gather_kernel_matches_reference_on_assembled_storage(tmp_path):
    rvn = _load("rvn_ple_storage_gpu", _MODULE_REL)
    ple = _load("packed_ple_gpu", _PLE_REL)
    root, packed, scales = _checkpoint(tmp_path)
    storage = rvn.load_for_checkpoint(
        str(root), storage=ple.PackedPLEStorage(ROWS, COLS, pin_memory=False))
    assert torch.equal(storage.weight, packed)

    device = "cuda"
    weight = storage.weight.to(device)
    scales_gpu = storage.scales.to(device)
    ids = torch.tensor([2, 0, 3, 1], dtype=torch.int64, device=device)
    out = torch.empty((ids.numel(), COLS), dtype=torch.bfloat16, device=device)
    ple.gather_packed_kernel[(ids.numel(),)](
        weight.data_ptr(), scales_gpu.data_ptr(), ids, out,
        storage.global_scale, COLS, 0, ROWS, False, COLS,
        enable_fp_fusion=False)
    expected = rvn.dequant_reference(
        packed[[2, 0, 3, 1]], scales[[2, 0, 3, 1]], storage.global_scale)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

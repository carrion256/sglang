"""CPU tests for the shared pinned PLE host table (patch 0060).

The runtime target only exists inside the applied series: set
``RVN_PLE_TREE`` to the root of a tree ``patches/series.rvn-w4a16`` plus
``patches/0060-rvn-ple-shared-ple.patch`` were applied to with ``-p1`` (the
throwaway container does exactly this). No GPU is required: the
cudaHostRegister/cudaHostGetDevicePointer handshake is exercised through an
injected cudart shim, so the mapping, alias-check and fallback bookkeeping
all run on a CPU host. CPU-only torch wheels refuse pin_memory=True; an
autouse fixture drops the kwarg only when no accelerator backend exists, so
the call sites keep pin_memory=True and pin for real inside the container.
Sizing literals mirror docs/rvn-ple-storage-schema.md (dim divisible by 16,
half-byte packing, one e4m3 scale per 16 columns) independently of the
loader, per the tests/rvn_ple convention.
"""
import ctypes
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
_PLE_REL = Path("python/sglang/srt/models/packed_ple.py")

ROWS = 48
DIM = 64  # weight (48, 32) u8; scales (48, 4) e4m3
WEIGHT_SHAPE = (ROWS, DIM // 2)
SCALES_SHAPE = (ROWS, DIM // 16)


def _tree():
    tree = os.environ.get("RVN_PLE_TREE")
    if not tree:
        pytest.skip(
            "packed_ple.py ships only inside patches/series.rvn-w4a16; set "
            "RVN_PLE_TREE to the tree the series (incl. patch 0060) was "
            "applied to", allow_module_level=True)
    root = Path(tree)
    src = root / _PLE_REL
    assert src.is_file(), f"RVN_PLE_TREE={tree} lacks {_PLE_REL}"
    assert "_host_table" in src.read_text(), (
        f"RVN_PLE_TREE={tree} lacks patch 0060: apply "
        "patches/0060-rvn-ple-shared-ple.patch to that tree root")
    return root


TREE = _tree()


def _load_ple():
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        # Kernel module load only: the tests never launch a kernel, and
        # @triton.jit must stay an identity so tl annotations are inert.
        triton = types.ModuleType("triton")
        triton.jit = lambda fn=None, **kw: fn if fn is not None else (
            lambda f: f)
        tl = types.ModuleType("triton.language")
        tl.constexpr = object
        triton.language = tl
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = tl
    spec = importlib.util.spec_from_file_location(
        "packed_ple_0060", TREE / _PLE_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ple = _load_ple()


# ------------------------------------------------------------- cudart shims


class _FakeCudart:
    def cudaHostRegister(self, host, nbytes, flags):
        return 0

    def cudaHostUnregister(self, host):
        return 0


class _FakeLib:
    """cudaHostGetDevicePointer shim; alias None mirrors the host pointer."""

    def __init__(self, alias=None):
        self.alias = alias
        self.calls = 0

    def cudaHostGetDevicePointer(self, device_ref, host_ref, flags):
        self.calls += 1
        device_ref._obj.value = (self.alias if self.alias is not None
                                 else host_ref.value)
        return 0


@pytest.fixture
def cudart_shim(monkeypatch):
    fake, lib = _FakeCudart(), _FakeLib()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: lib)
    return lib


@pytest.fixture(autouse=True)
def _pinned_allocator(monkeypatch):
    # CPU-only torch wheels refuse pin_memory=True; the battery container's
    # CUDA torch never does. Emulate the missing allocator only when absent,
    # keeping pin_memory=True at every call site.
    if torch.cuda.is_available():
        return
    orig = torch.empty

    def empty(*args, **kw):
        if kw.get("pin_memory"):
            kw = dict(kw, pin_memory=False)
        return orig(*args, **kw)

    monkeypatch.setattr(torch, "empty", empty)


@pytest.fixture
def rank0(monkeypatch):
    """Deterministic shard key: no process group, no launcher $RANK."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def _new():
    return ple.PackedPLEStorage(ROWS, DIM)


# ------------------------------------------------------------------- tests


def test_env_unset_keeps_private_pinned(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_PLE_SHARED_DIR", raising=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    before = len(ple._shared_maps)
    st = _new()
    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert st.weight.dtype is torch.uint8
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.scales.dtype is torch.float8_e4m3fn
    assert list(shared.iterdir()) == []
    assert len(ple._shared_maps) == before


def test_shared_dir_two_instances_one_backing(tmp_path, monkeypatch,
                                               cudart_shim, rank0):
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    before = len(ple._shared_maps)
    a = _new()
    a.weight.fill_(0xA5)
    a.scales.view(torch.uint8).fill_(0x5A)
    b = _new()  # second attach: sees the bytes, does not rewrite or zero
    assert sorted(p.name for p in shared.iterdir()) == [
        "ple-packed-rows-rank0-48x32-uint8.bin",
        "ple-scales-rank0-48x4-float8_e4m3fn.bin"]
    rows_file = shared / "ple-packed-rows-rank0-48x32-uint8.bin"
    scales_file = shared / "ple-scales-rank0-48x4-float8_e4m3fn.bin"
    assert rows_file.stat().st_size == ROWS * (DIM // 2)
    assert scales_file.stat().st_size == ROWS * (DIM // 16)
    assert torch.equal(b.weight, torch.full_like(b.weight, 0xA5))
    assert torch.equal(b.scales.view(torch.uint8),
                       torch.full_like(b.scales.view(torch.uint8), 0x5A))
    # the backing files, not copies, hold the bytes
    assert rows_file.read_bytes() == bytes([0xA5]) * rows_file.stat().st_size
    assert scales_file.read_bytes() == bytes([0x5A]) * scales_file.stat().st_size
    assert len(ple._shared_maps) == before + 4
    assert cudart_shim.calls == 4  # alias checked for every table


def test_device_alias_mismatch_falls_back(tmp_path, monkeypatch, capsys):
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    lib = _FakeLib(alias=0xDEADBEEF)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudart())
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: lib)
    before = len(ple._shared_maps)
    st = _new()
    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert len(ple._shared_maps) == before  # mismatched maps are not kept
    assert lib.calls == 2
    err = capsys.readouterr().err
    assert err.count("device alias") == 2
    assert err.count("falling back to private pinned table") == 2


def test_unusable_shared_dir_falls_back_private(tmp_path, monkeypatch,
                                                capsys, cudart_shim):
    # shared dir under an existing regular file: makedirs fails for every
    # uid, including root, so the injection cannot flake on privileges.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(blocker / "shared"))
    before = len(ple._shared_maps)
    st = _new()  # must never fail boot
    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before
    assert cudart_shim.calls == 0  # failed before any registration
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


def test_tp2_ranks_do_not_share_mapping(tmp_path, monkeypatch, cudart_shim,
                                        rank0):
    # TP2 ranks hold the SAME padded shard shape but DIFFERENT rows: the
    # role+shape+dtype key alone would make them corrupt each other.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    a = _new()  # shard 0
    a.weight.fill_(0xA5)
    monkeypatch.setenv("RANK", "1")  # sibling rank, same shape and role
    b = _new()
    assert sorted(p.name for p in shared.iterdir()) == [
        "ple-packed-rows-rank0-48x32-uint8.bin",
        "ple-packed-rows-rank1-48x32-uint8.bin",
        "ple-scales-rank0-48x4-float8_e4m3fn.bin",
        "ple-scales-rank1-48x4-float8_e4m3fn.bin"]
    assert torch.equal(a.weight, torch.full_like(a.weight, 0xA5))  # not clobbered
    b.weight.fill_(0x3C)
    assert torch.equal(a.weight, torch.full_like(a.weight, 0xA5))  # isolated
    assert torch.equal(b.weight, torch.full_like(b.weight, 0x3C))


def test_dist_rank_shares_the_env_rank_namespace(tmp_path, monkeypatch,
                                                 cudart_shim, rank0):
    # sglang TP2 workers learn their rank through torch.distributed, not
    # necessarily through $RANK; both sources must key one namespace.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    st = _new()
    assert (shared / "ple-packed-rows-rank1-48x32-uint8.bin").is_file()
    assert not (shared / "ple-packed-rows-rank0-48x32-uint8.bin").exists()
    assert tuple(st.weight.shape) == WEIGHT_SHAPE

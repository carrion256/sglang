"""CPU tests for the shared pinned PLE host table (patches 0060, 0061).

The runtime target only exists inside the applied series: set
``RVN_PLE_TREE`` to the root of a tree ``patches/series.rvn-w4a16``,
``patches/0060-rvn-ple-shared-ple.patch`` and
``patches/0061-rvn-shared-ple-identity-cleanup.patch`` were applied to with
``-p1`` (the throwaway container does exactly this). No GPU is required: the
cudaHostRegister/cudaHostGetDevicePointer handshake is exercised through an
injected cudart shim, and the runtime lookup is repointed at a maps file this
suite writes, so the checkpoint-identity key, the foreign-table refusal, the
alias check and the unpin-before-close bookkeeping all run on a CPU host. CPU
only torch wheels refuse pin_memory=True; an autouse fixture drops the kwarg
only when no accelerator backend exists, so the call sites keep pin_memory=True
and pin for real inside the container. Sizing literals mirror
docs/rvn-ple-storage-schema.md (dim divisible by 16, half-byte packing, one
e4m3 scale per 16 columns) independently of the loader, per the tests/rvn_ple
convention.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_PLE_REL = Path("python/sglang/srt/models/packed_ple.py")

ROWS = 48
DIM = 64  # weight (48, 32) u8; scales (48, 4) e4m3
WEIGHT_SHAPE = (ROWS, DIM // 2)
SCALES_SHAPE = (ROWS, DIM // 16)

# A table size past 2**32, which is the production problem in miniature: the
# served rows table is 25,600,122,880 bytes. A size that does not fit a C int
# has to reach the runtime intact or the pin is not the pin that was asked for,
# and a 48x32 fixture cannot tell the difference.
BIG_ROWS = (2 ** 32 + 512) // (DIM // 2)
BIG_WEIGHT_BYTES = BIG_ROWS * (DIM // 2)
BIG_SCALES_BYTES = BIG_ROWS * (DIM // 16)

STAMP = {"source": "/models/source-checkpoint",
         "encoder_version": "rvn-mtp-graft-r1", "count": 1}

CUDART = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13"
OTHER_CUDART = "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13.0.96"
ONE_RUNTIME = (("00:53", "43543918"), CUDART)
MAP_LINE = "7f0000000000-7f0000001000 r-xp 00000000 {} {}  {}".format

# --------------------------------------------------------------- the injected
# runtime. Stands in for the handle the module resolves so the handshake can
# run on a CPU host, and keeps the evidence the module's own guarantees are
# checked against: what it was asked to pin, and whether a mapping was still
# open at the moment its pin was released.


class _InjectedCall:
    """One entry point of the injected runtime; records its signature."""

    def __init__(self, owner, name):
        self._owner = owner
        self._name = name

    def _declare(self, kind, value):
        self._owner.declared[(self._name, kind)] = value
        self._owner.log.append(("declare", self._name, kind))

    @property
    def argtypes(self):
        return self._owner.declared.get((self._name, "argtypes"))

    @argtypes.setter
    def argtypes(self, value):
        self._declare("argtypes", tuple(value))

    @property
    def restype(self):
        return self._owner.declared.get((self._name, "restype"))

    @restype.setter
    def restype(self, value):
        self._declare("restype", value)

    def __call__(self, *args):
        return self._owner.dispatch(self._name, args)


def _address(value):
    """The pointer a call argument carries, for the record the shim keeps."""
    if isinstance(value, int):
        return value
    if isinstance(value, ctypes.c_void_p):
        return value.value
    target = getattr(value, "_obj", None)  # a byref() wrapper
    if target is not None:
        return ctypes.addressof(target)
    return None


def _cudart_opens(runtime):
    """Every request to open a CUDA runtime that this suite recorded.

    Scoped to the CUDA runtime on purpose: torch and torchvision load their own
    libraries during a private fallback, and a check that counted all of them
    would trip on unrelated imports instead of on the thing it guards -- an
    attempt to reach a cudart handle before a checkpoint identity resolves, or
    by a soname the linker could answer with somebody else's library.
    """
    return [(path, mode) for path, mode in runtime.cdll_calls
            if "libcudart" in str(path)]

class _FakeRuntime:
    """cudaHostRegister/cudaHostUnregister/cudaHostGetDevicePointer shim.

    ``alias`` None mirrors the host pointer the way a device with unified
    addressing answers; anything else is the mismatch the module must refuse.
    Every call is recorded, and a size past 2**31 is refused unless the
    signatures that marshal it as a 64-bit size_t were declared on this handle
    first -- dropping those declarations degrades the shared path silently in
    production, and fails loudly here.
    """

    def __init__(self, alias=None, fail_register=False):
        self.alias = alias
        self.fail_register = fail_register
        self.log = []
        self.declared = {}
        self.pins = []
        self.unpins = []
        self.mapping_live_at_unpin = []
        self.cdll_calls = []
        self.cudaHostRegister = _InjectedCall(self, "cudaHostRegister")
        self.cudaHostUnregister = _InjectedCall(self, "cudaHostUnregister")
        self.cudaHostGetDevicePointer = _InjectedCall(
            self, "cudaHostGetDevicePointer")

    @property
    def alias_calls(self):
        return sum(1 for row in self.log
                   if row[0] == "call" and row[1] == "cudaHostGetDevicePointer")

    def dispatch(self, name, args):
        self.log.append(("call", name) + tuple(_address(a) for a in args))
        if name == "cudaHostRegister":
            host, nbytes, flags = args
            if int(nbytes) > 2 ** 31 and not self._register_signature_ok():
                raise AssertionError(
                    f"cudaHostRegister asked to pin {nbytes} bytes with no "
                    "declared 64-bit signature; that size cannot survive the "
                    "call")
            if self.fail_register:
                return 17  # cudaErrorMemoryAllocation
            self.pins.append((_address(host), int(nbytes), int(flags)))
            return 0
        if name == "cudaHostGetDevicePointer":
            device_ref, host_ref, _flags = args
            device_ref._obj.value = (self.alias if self.alias is not None
                                     else _address(host_ref))
            return 0
        if name == "cudaHostUnregister":
            host = _address(args[0])
            self.unpins.append(host)
            self.mapping_live_at_unpin.append(_mapping_live(host))
            return 0
        raise AssertionError(f"unexpected runtime call {name}")

    def _register_signature_ok(self):
        return (self.declared.get(("cudaHostRegister", "argtypes"))
                == (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
                and self.declared.get(("cudaHostRegister", "restype"))
                is ctypes.c_int)


def _mapping_live(address):
    """True when ``address`` sits inside a mapping of this very process.

    Reads the real /proc/self/maps, not the seam the tests repoint, so it can
    say whether a mapping was still open when its pin was handed back. A pin
    released after the mapping was closed lands on memory nothing owns.
    """
    if address is None:
        return False
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
        for line in maps:
            fields = line.split()
            if not fields or "-" not in fields[0]:
                continue
            try:
                low, high = (int(bound, 16) for bound in fields[0].split("-"))
            except ValueError:
                continue
            if low <= address < high:
                return True
    return False


def _install(tmp_path, monkeypatch, runtime, entries=(ONE_RUNTIME,)):
    """Repoint the module's runtime lookup at a maps file this suite owns.

    The path it reports is the only thing the module may hand to CDLL, and the
    shim is what comes back, so nothing in these tests ever binds a real library
    or reaches a GPU.
    """
    maps = tmp_path / "fake-maps"
    maps.write_text("".join(
        MAP_LINE(ident[0], ident[1], path) + "\n" for ident, path in entries))
    monkeypatch.setattr(ple, "_PROC_MAPS", str(maps))

    def cdll(path, *args, **kwargs):
        runtime.cdll_calls.append((path, kwargs.get("mode")))
        return runtime

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    return runtime


@pytest.fixture
def cudart_shim(tmp_path, monkeypatch):
    """A resolvable, injected runtime, for the paths that must pin and check."""
    return _install(tmp_path, monkeypatch, _FakeRuntime())


@pytest.fixture
def maps_seam(tmp_path, monkeypatch):
    """Install a chosen map and shim; hands back the injected runtime."""

    def install(entries, runtime=None):
        return _install(tmp_path, monkeypatch, runtime or _FakeRuntime(),
                        entries)

    return install


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


@pytest.fixture
def serving_identity(tmp_path, monkeypatch):
    """A grafted checkpoint plus the serving config that points at it.

    Stands in for a serving process: sglang's own accessor is what hands the
    weight loader the model path, and the checkpoint directory holds the
    rvn_mtp_graft stamp patch 0058 gates a grafted draft on. The tag the shared
    names are keyed by is recomputed here from that recipe alone.
    """
    model_path = tmp_path / "grafted-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForCausalLM"],
        "mtp_num_hidden_layers": 1,
        "rvn_mtp_graft": dict(STAMP),
    }))
    return _serve_from(tmp_path, monkeypatch, model_path, STAMP)


def _serve_from(tmp_path, monkeypatch, model_path, stamp):
    """Point the serving accessor at ``model_path`` and return its tag.

    The accessor is installed as a stub module at the import boundary rather
    than by importing the image's own sglang: importing
    ``sglang.srt.server_args`` inside a pytest process aborts the interpreter in
    this image (its ``ant_data_rw_api`` Qt layer refuses to mix worker and main
    threads), which is a fault of the image, not of the code under test. A
    serving process reaches the model path through exactly one import, and this
    is where that import is answered, so the tag under test is still derived
    from a served model path and a checkpoint's own stamp.
    """
    stub = types.ModuleType("sglang.srt.server_args")
    stub.get_global_server_args = lambda: types.SimpleNamespace(
        model_path=str(model_path))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", stub)
    return _expected_tag(str(model_path), stamp)


def _tree():
    tree = os.environ.get("RVN_PLE_TREE")
    if not tree:
        pytest.skip(
            "packed_ple.py ships only inside patches/series.rvn-w4a16; set "
            "RVN_PLE_TREE to the tree the series (incl. patches 0060 and 0061) "
            "was applied to", allow_module_level=True)
    root = Path(tree)
    src = root / _PLE_REL
    assert src.is_file(), f"RVN_PLE_TREE={tree} lacks {_PLE_REL}"
    text = src.read_text()
    assert "_host_table" in text, (
        f"RVN_PLE_TREE={tree} lacks patch 0060: apply "
        "patches/0060-rvn-ple-shared-ple.patch to that tree root")
    assert "_mapped_cudart" in text and "cudaHostRegister.argtypes" in text, (
        f"RVN_PLE_TREE={tree} lacks patch 0061: apply "
        "patches/0061-rvn-shared-ple-identity-cleanup.patch to that tree root")
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
        "packed_ple_0061", TREE / _PLE_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ple = _load_ple()


def _expected_tag(model_path, stamp):
    """The shared-name tag, derived here from the stamp recipe alone.

    Computed independently of packed_ple so the names under test are checked
    against the identity a checkpoint actually has.
    """
    identity = "\n".join((
        "rvn-ple-shared-identity-v1",
        f"model={model_path}",
        f"graft_source={stamp['source']}",
        f"graft_encoder_version={stamp['encoder_version']}",
        f"graft_count={stamp['count']}"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _new(rows=ROWS, dim=DIM):
    return ple.PackedPLEStorage(rows, dim)


def _names(shared):
    return sorted(entry.name for entry in os.scandir(shared))


def _tagged(tag, rank="0", shape=WEIGHT_SHAPE, dtype="uint8"):
    return (f"ple-{'packed-rows' if dtype == 'uint8' else 'scales'}"
            f"-rank{rank}-{shape[0]}x{shape[1]}-{dtype}-{tag}.bin")


def _table_files(tag, rank="0"):
    return [_tagged(tag, rank),
            _tagged(tag, rank, SCALES_SHAPE, "float8_e4m3fn")]


# ------------------------------------------------------------------ baseline


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


# ------------------------------------------------- identity keys the shared name


def test_shared_tables_are_keyed_to_the_checkpoint_tag(tmp_path, monkeypatch,
                                                       cudart_shim, rank0,
                                                       serving_identity):
    tag = serving_identity
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    before = len(ple._shared_maps)

    def no_torch_cudart():
        raise AssertionError("the shared pin must not go through torch.cuda")

    monkeypatch.setattr(torch.cuda, "cudart", no_torch_cudart)
    a = _new()
    a.weight.fill_(0xA5)
    a.scales.view(torch.uint8).fill_(0x5A)
    b = _new()  # second attach: sees the bytes, does not rewrite or zero

    assert _names(shared) == sorted(_table_files(tag))
    rows_file = shared / _tagged(tag)
    scales_file = shared / _tagged(tag, "0", SCALES_SHAPE, "float8_e4m3fn")
    assert rows_file.stat().st_size == ROWS * (DIM // 2)
    assert scales_file.stat().st_size == ROWS * (DIM // 16)
    assert torch.equal(b.weight, torch.full_like(b.weight, 0xA5))
    assert torch.equal(b.scales.view(torch.uint8),
                       torch.full_like(b.scales.view(torch.uint8), 0x5A))
    # the backing files, not copies, hold the bytes
    assert rows_file.read_bytes() == bytes([0xA5]) * rows_file.stat().st_size
    assert scales_file.read_bytes() == bytes([0x5A]) * scales_file.stat().st_size
    assert len(ple._shared_maps) == before + 4
    assert cudart_shim.alias_calls == 4  # alias checked for every table
    # the handle is this process's mapped runtime and it is asked for locally:
    # no bare soname, no toolkit copy, no torch-internal cudart anywhere
    assert cudart_shim.cdll_calls == [(CUDART, ctypes.RTLD_LOCAL)] * 4
    assert cudart_shim.log[0] == ("declare", "cudaHostRegister", "argtypes")
    assert all(row[0] == "declare" for row in cudart_shim.log[:6])
    assert len(cudart_shim.pins) == 4 and all(
        flags == 3 and nbytes in (ROWS * (DIM // 2), ROWS * (DIM // 16))
        for _host, nbytes, flags in cudart_shim.pins)


def test_large_table_size_reaches_the_runtime_intact(tmp_path, monkeypatch,
                                                     maps_seam, rank0,
                                                     serving_identity):
    tag = serving_identity
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME])
    before = len(ple._shared_maps)

    st = _new(BIG_ROWS, DIM)

    assert BIG_WEIGHT_BYTES > 2 ** 32
    assert tuple(st.weight.shape) == (BIG_ROWS, DIM // 2)
    assert tuple(st.scales.shape) == (BIG_ROWS, DIM // 16)
    assert _names(shared) == sorted((
        _tagged(tag, "0", (BIG_ROWS, DIM // 2)),
        _tagged(tag, "0", (BIG_ROWS, DIM // 16), "float8_e4m3fn")))
    assert (shared / _tagged(tag, "0", (BIG_ROWS, DIM // 2))).stat().st_size \
        == BIG_WEIGHT_BYTES
    assert (shared / _tagged(tag, "0", (BIG_ROWS, DIM // 16),
                             "float8_e4m3fn")).stat().st_size \
        == BIG_SCALES_BYTES
    # the signatures that keep the byte count 64-bit, declared before any call
    assert runtime.declared[("cudaHostRegister", "argtypes")] == (
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    assert runtime.declared[("cudaHostRegister", "restype")] is ctypes.c_int
    assert runtime.declared[("cudaHostUnregister", "argtypes")] == (
        ctypes.c_void_p,)
    assert runtime.declared[("cudaHostUnregister", "restype")] is ctypes.c_int
    assert runtime.declared[("cudaHostGetDevicePointer", "argtypes")] == (
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint)
    assert runtime.declared[("cudaHostGetDevicePointer", "restype")] \
        is ctypes.c_int
    assert all(row[0] == "declare" for row in runtime.log[:6])
    assert sorted(nbytes for _h, nbytes, _f in runtime.pins) == sorted(
        [BIG_WEIGHT_BYTES, BIG_SCALES_BYTES])
    assert len(runtime.pins) == 2
    assert runtime.unpins == []
    assert runtime.alias_calls == 2
    assert len(ple._shared_maps) == before + 2
    assert runtime.cdll_calls == [(CUDART, ctypes.RTLD_LOCAL)] * 2


# -------------------------------------------------------- foreign tables


def test_unknown_checkpoint_identity_declines_shared(tmp_path, monkeypatch,
                                                     maps_seam, rank0, capsys):
    # No serving config, so no checkpoint to key a shared table by: the shared
    # path is declined outright rather than leaving an anonymous table that the
    # next same-shape checkpoint would walk straight into.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME])
    before = len(ple._shared_maps)
    st = _new()  # must never fail boot

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert not shared.exists()
    assert len(ple._shared_maps) == before
    assert runtime.pins == [] and runtime.alias_calls == 0
    assert _cudart_opens(runtime) == []  # declined before any handle was opened
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2

@pytest.mark.parametrize("write_config", [False, True])
def test_unstamped_checkpoint_declines_shared(tmp_path, monkeypatch,
                                              maps_seam, rank0, capsys,
                                              write_config):
    # A stamp the serving checkpoint itself carries is the identity; a path on
    # its own is not. Whether the config is missing or present but unstamped,
    # hashing the path anyway would hand out a tag that looks like an identity
    # while surviving the checkpoint being swapped underneath -- the exact
    # collision this gate exists to close -- so the shared path is declined.
    model_path = tmp_path / "served-checkpoint"
    model_path.mkdir()
    if write_config:
        (model_path / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForCausalLM"],
            "mtp_num_hidden_layers": 1,
        }))
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME])
    import sys as _sys
    import types as _types
    stub = _types.ModuleType("sglang.srt.server_args")
    stub.get_global_server_args = lambda: _types.SimpleNamespace(
        model_path=str(model_path))
    monkeypatch.setitem(_sys.modules, "sglang.srt.server_args", stub)
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert not shared.exists()
    assert len(ple._shared_maps) == before
    assert runtime.pins == [] and runtime.alias_calls == 0
    assert _cudart_opens(runtime) == []
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


@pytest.mark.parametrize("name", [
    # a table of this role, rank, shape and dtype keyed to no identity: the
    # naming this shipped with before patch 0061
    "ple-packed-rows-rank0-48x32-uint8.bin",
    # ... and one keyed to somebody else's checkpoint
    "ple-packed-rows-rank0-48x32-uint8-0123456789abcdef.bin",
])
def test_foreign_table_is_refused_and_left_alone(tmp_path, monkeypatch,
                                                 maps_seam, rank0, capsys,
                                                 name, serving_identity):
    shared = tmp_path / "shared"
    shared.mkdir()
    foreign = shared / name
    foreign.write_bytes(b"\xee" * 128)
    untouched = (foreign.read_bytes(), foreign.stat().st_size,
                 foreign.stat().st_mtime_ns)
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME])
    before = len(ple._shared_maps)

    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    # refused without being opened, truncated, rewritten or unlinked
    assert (foreign.read_bytes(), foreign.stat().st_size,
            foreign.stat().st_mtime_ns) == untouched
    assert _names(shared) == [name]
    assert len(ple._shared_maps) == before
    assert runtime.pins == [] and runtime.alias_calls == 0
    assert _cudart_opens(runtime) == []  # refused before any handle was opened
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2
    assert name in err


# ------------------------------------------------- alias check and clean-up


def test_device_alias_mismatch_unpins_before_it_closes(tmp_path, monkeypatch,
                                                       maps_seam, rank0, capsys,
                                                       serving_identity):
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME], _FakeRuntime(alias=0xDEADBEEF))
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert len(ple._shared_maps) == before  # mismatched maps are not kept
    assert runtime.alias_calls == 2
    # every pin that was taken is given back once, through the same handle, and
    # while the mapping it lives in is still open -- never after it is closed
    assert runtime.unpins == [host for host, _n, _f in runtime.pins]
    assert len(runtime.unpins) == 2
    assert runtime.mapping_live_at_unpin == [True, True]
    err = capsys.readouterr().err
    assert err.count("device alias") == 2
    assert err.count("falling back to private pinned table") == 2


def test_registration_failure_is_not_unregistered(tmp_path, monkeypatch,
                                                  maps_seam, rank0, capsys,
                                                  serving_identity):
    # Nothing was pinned, so nothing may be unregistered: an unpin of a pin that
    # was never taken is as much a corruption as a missing one.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME], _FakeRuntime(fail_register=True))
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before
    assert runtime.pins == []
    assert runtime.unpins == []
    assert runtime.alias_calls == 0
    err = capsys.readouterr().err
    assert err.count("cudaHostRegister failed") == 2
    assert err.count("falling back to private pinned table") == 2


def test_failure_after_registration_still_unpins(tmp_path, monkeypatch,
                                                 maps_seam, rank0, capsys,
                                                 serving_identity):
    # A runtime that takes the pin and then cannot answer the alias query is a
    # runtime in trouble, and the mapping it registered still has to be
    # released before anything closes it.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME])

    def broken_alias(*args):
        runtime.log.append(("call", "cudaHostGetDevicePointer"))
        raise OSError("alias query died")

    runtime.cudaHostGetDevicePointer = broken_alias
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before
    assert runtime.unpins == [host for host, _n, _f in runtime.pins]
    assert len(runtime.unpins) == 2
    assert runtime.mapping_live_at_unpin == [True, True]
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


# ------------------------------------------------------ runtime resolution


def test_one_runtime_under_two_names_is_still_one(tmp_path, monkeypatch,
                                                  maps_seam, rank0,
                                                  serving_identity):
    # The same library can be mapped under more than one name. That is one
    # runtime, and refusing it would cost the node a second pinned table for
    # nothing, so the handle has to come from the map either way.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME,
                         (("00:53", "43543918"), "/opt/alias/libcudart.so.13")])
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert _names(shared) == sorted(_table_files(serving_identity))
    assert [path for path, _mode in runtime.cdll_calls] == [CUDART] * 2
    assert {mode for _path, mode in runtime.cdll_calls} == {ctypes.RTLD_LOCAL}
    assert runtime.alias_calls == 2
    assert st.weight.is_cuda is False


def test_two_mapped_runtimes_decline_shared(tmp_path, monkeypatch, maps_seam,
                                            rank0, capsys, serving_identity):
    # Two distinct libcudart files. Whichever one took the pin, the alias answer
    # could come from the other, and the shared table would be keyed to a check
    # that never happened against the runtime that matters.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([ONE_RUNTIME, (("00:53", "43543999"), OTHER_CUDART)])
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before
    assert not shared.exists() or _names(shared) == []
    assert _cudart_opens(runtime) == []  # declined before any handle was opened
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


def test_no_mapped_runtime_declines_shared(tmp_path, monkeypatch, maps_seam,
                                           rank0, capsys, serving_identity):
    # Never a CDLL("libcudart.so") guess: with no runtime mapped the shared
    # mapping is declined, and the private pinned table is what gets allocated.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    runtime = maps_seam([(("00:53", "43543918"),
                          "/usr/lib/x86_64-linux-gnu/libz.so.1")])
    before = len(ple._shared_maps)
    st = _new()

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert len(ple._shared_maps) == before
    assert not shared.exists() or _names(shared) == []
    assert _cudart_opens(runtime) == []  # declined: never a guessed soname
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


def test_unusable_shared_dir_falls_back_private(tmp_path, monkeypatch,
                                                maps_seam, rank0, capsys,
                                                serving_identity):
    # A shared dir under an existing regular file: makedirs fails for every uid,
    # including root, so the injection cannot flake on privileges.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(blocker / "shared"))
    runtime = maps_seam([ONE_RUNTIME])
    before = len(ple._shared_maps)
    st = _new()  # must never fail boot

    assert tuple(st.weight.shape) == WEIGHT_SHAPE
    assert tuple(st.scales.shape) == SCALES_SHAPE
    assert st.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before
    assert runtime.pins == [] and runtime.alias_calls == 0
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2


# ------------------------------------------------------------- rank keying


def test_tp2_ranks_do_not_share_mapping(tmp_path, monkeypatch, cudart_shim,
                                        rank0, serving_identity):
    # TP2 ranks hold the SAME padded shard shape but DIFFERENT rows: the
    # role+shape+dtype key alone would make them corrupt each other.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    a = _new()  # shard 0
    a.weight.fill_(0xA5)
    monkeypatch.setenv("RANK", "1")  # sibling rank, same shape and role
    b = _new()
    assert _names(shared) == sorted(
        _table_files(serving_identity) + _table_files(serving_identity, "1"))
    assert torch.equal(a.weight, torch.full_like(a.weight, 0xA5))  # not clobbered
    b.weight.fill_(0x3C)
    assert torch.equal(a.weight, torch.full_like(a.weight, 0xA5))  # isolated
    assert torch.equal(b.weight, torch.full_like(b.weight, 0x3C))


def test_dist_rank_shares_the_env_rank_namespace(tmp_path, monkeypatch,
                                                 cudart_shim, rank0,
                                                 serving_identity):
    # sglang TP2 workers learn their rank through torch.distributed, not
    # necessarily through $RANK; both sources must key one namespace.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    st = _new()
    assert (shared / _tagged(serving_identity, "1")).is_file()
    assert not (shared / _tagged(serving_identity, "0")).exists()
    assert tuple(st.weight.shape) == WEIGHT_SHAPE


def test_second_checkpoint_of_same_shape_is_refused_the_shared_dir(tmp_path,
                                                                  monkeypatch,
                                                                  cudart_shim,
                                                                  rank0,
                                                                  serving_identity,
                                                                  capsys):
    # The collision patch 0061 exists to close: same architecture, same padded
    # shard shape, a different checkpoint. Under patch 0060 this is exactly the
    # case that silently rewrote a live replica's table. The stranger is turned
    # away, and the table it came for is left byte-for-byte as it was.
    shared = tmp_path / "shared"
    monkeypatch.setenv("SGLANG_PLE_SHARED_DIR", str(shared))
    a = _new()
    a.weight.fill_(0xA5)
    rows_file = shared / _tagged(serving_identity)
    original = rows_file.read_bytes()
    assert original == bytes([0xA5]) * rows_file.stat().st_size
    pins_before = len(cudart_shim.pins)
    before = len(ple._shared_maps)

    other = tmp_path / "other-checkpoint"
    other.mkdir()
    other_stamp = dict(STAMP, source="/models/a-different-source-checkpoint")
    (other / "config.json").write_text(json.dumps({"rvn_mtp_graft": other_stamp}))
    assert _serve_from(tmp_path, monkeypatch, other, other_stamp) \
        != serving_identity
    b = _new()

    assert tuple(b.weight.shape) == WEIGHT_SHAPE
    assert tuple(b.scales.shape) == SCALES_SHAPE
    assert b.weight.device.type == "cpu"
    assert len(ple._shared_maps) == before  # nothing new is mapped or shared
    assert len(cudart_shim.pins) == pins_before  # and nothing was pinned
    assert _names(shared) == sorted(_table_files(serving_identity))  # no new file
    b.weight.fill_(0x3C)  # the private table, wherever it lives
    assert rows_file.read_bytes() == original  # still not the stranger's target
    err = capsys.readouterr().err
    assert err.count("falling back to private pinned table") == 2

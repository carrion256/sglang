"""PLE checkpoint correctness on CPU fixtures and real CUDA transfers."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

if os.environ.get("QWEN_HICACHE_TEST_DEVICE", "cpu") == "cpu":
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()

from sglang.srt.mem_cache import memory_pool_host as host_module
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, MambaPoolHost
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool


@pytest.fixture
def device():
    d = torch.device(os.environ.get("QWEN_HICACHE_TEST_DEVICE", "cpu"))
    if d.type == "cuda":
        assert torch.cuda.is_available(), "Requested CUDA tests must run, not skip"
    return d


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def io_indices(indices, backend):
    # HiCacheController.move_indices keeps direct-copy indices on the CPU.
    return indices.cpu() if backend == "direct" else indices


def pattern(shape, dtype, device, offset=0):
    count = 1
    for n in shape:
        count *= n
    return ((torch.arange(count, device=device) % 97) + offset).to(dtype).reshape(shape)


def make_pool(device, companions=True):
    pool = MambaPool.__new__(MambaPool)
    pool.size = 7
    pool.num_mamba_layers = 3
    pool.mamba_layer_ids = [0, 2, 3]
    pool.device = device.type
    pool.mamba_cache = MambaPool.State(
        conv=[pattern((3, 8, 12, 4), torch.bfloat16, device, 2)],
        temporal=pattern((3, 8, 2, 8, 8), torch.bfloat16, device, 4),
    )
    conv = ShortConvPool.__new__(ShortConvPool)
    conv.conv_state = pattern((2, 8, 8, 4), torch.bfloat16, device, 6)
    conv.layer_map = {1: 0, 3: 1}
    ngram = NGramPool.__new__(NGramPool)
    ngram.context = pattern((8, 2), torch.int64, device, 1000)
    pool._slot_siblings = [conv, ngram] if companions else []
    pool.replayssm_cache_base = None
    return pool


def tensors(pool):
    result = {"conv": pool.mamba_cache.conv[0], "temporal": pool.mamba_cache.temporal}
    if pool._slot_siblings:
        result["ple_conv"] = pool._slot_siblings[0].conv_state
        result["ple_ngram"] = pool._slot_siblings[1].context.unsqueeze(0)
    return result


def snapshot(pool, indices):
    return {name: tensor[:, indices].clone() for name, tensor in tensors(pool).items()}


def poison(pool, indices):
    for i, tensor in enumerate(tensors(pool).values(), 1):
        tensor[:, indices] = -100 * i


def assert_state(pool, indices, expected):
    for name, tensor in tensors(pool).items():
        assert torch.equal(tensor[:, indices], expected[name]), name


@pytest.fixture(autouse=True)
def cpu_transport(monkeypatch, device):
    if device.type != "cpu":
        return

    def backup(*, src_layers, dst, src_indices, dst_indices, **kwargs):
        dst[dst_indices, :, 0] = src_layers[:, src_indices].transpose(0, 1)

    def restore(*, src, dst, src_indices, dst_indices, layer_id, **kwargs):
        dst[dst_indices] = src[src_indices, layer_id, 0]

    # Retain real construction, state selection, allocation and lifecycle.
    # CUDA runs do not replace either transport function.
    monkeypatch.setattr(
        MambaPoolHost, "_copy_tensor_all_layers_lf_pf", staticmethod(backup)
    )
    monkeypatch.setattr(MambaPoolHost, "_copy_tensor_pf_lf", staticmethod(restore))


@pytest.fixture
def host_factory(device):
    made = []

    def create(pool, layout="page_first", **kwargs):
        host = MambaPoolHost(
            pool,
            host_to_device_ratio=2,
            host_size=kwargs.pop("host_size", 0),
            layout=layout,
            pin_memory=device.type == "cuda",
            **kwargs,
        )
        made.append(host)
        return host

    yield create
    sync(device)
    for host in made:
        host.destroy()


@pytest.mark.parametrize(
    "layout,backend", [("page_first", "kernel"), ("page_first_direct", "direct")]
)
@pytest.mark.parametrize("companions", [True, False])
@pytest.mark.parametrize("destination", [[1, 2], [4, 5]])
def test_complete_state_roundtrip(
    device, host_factory, layout, backend, companions, destination
):
    pool = make_pool(device, companions)
    host = host_factory(pool, layout)
    src = torch.tensor([1, 2], device=device)
    dst = torch.tensor(destination, device=device)
    rows = host.alloc(2)
    expected = snapshot(pool, src)
    host.backup_from_device_all_layer(pool, rows, io_indices(src, backend), backend)
    sync(device)
    poison(pool, dst)
    for layer in range(pool.num_mamba_layers):
        host.load_to_device_per_layer(
            pool, rows, io_indices(dst, backend), layer, backend
        )
    sync(device)
    assert_state(pool, dst, expected)


@pytest.mark.parametrize(
    "layout,backend", [("page_first", "kernel"), ("page_first_direct", "direct")]
)
def test_one_host_checkpoint_restores_tree_and_request_slots(
    device, host_factory, layout, backend
):
    pool = make_pool(device)
    host = host_factory(pool, layout)
    src = torch.tensor([1], device=device)
    dst = torch.tensor([4, 5], device=device)
    rows = host.alloc(1)
    expected = {
        k: v.repeat(1, 2, *([1] * (v.ndim - 2))) for k, v in snapshot(pool, src).items()
    }
    host.backup_from_device_all_layer(pool, rows, io_indices(src, backend), backend)
    sync(device)
    poison(pool, dst)
    for layer in range(pool.num_mamba_layers):
        host.load_to_device_per_layer(
            pool, rows.repeat(2), io_indices(dst, backend), layer, backend
        )
    sync(device)
    assert_state(pool, dst, expected)


@pytest.mark.parametrize("backend", ["kernel", "direct"])
def test_companions_ready_at_first_layer_event(device, host_factory, backend):
    pool = make_pool(device)
    host = host_factory(
        pool, "page_first" if backend == "kernel" else "page_first_direct"
    )
    src, dst = torch.tensor([1], device=device), torch.tensor([4], device=device)
    rows = host.alloc(1)
    expected = snapshot(pool, src)
    host.backup_from_device_all_layer(pool, rows, io_indices(src, backend), backend)
    sync(device)
    poison(pool, dst)
    host.load_to_device_per_layer(pool, rows, io_indices(dst, backend), 0, backend)
    sync(device)
    actual = snapshot(pool, dst)
    assert torch.equal(actual["ple_conv"], expected["ple_conv"])
    assert torch.equal(actual["ple_ngram"], expected["ple_ngram"])
    assert torch.equal(actual["temporal"][0], expected["temporal"][0])
    assert torch.all(actual["temporal"][1:] == -200)


@pytest.mark.parametrize("companions", [True, False])
def test_fixed_budget_counts_every_buffer(device, host_factory, companions):
    pool = make_pool(device, companions)
    budget = 30000
    host = host_factory(pool, host_size=budget / 1e9)
    allocated = sum(t.numel() * t.element_size() for t in host.get_hybrid_pool_buffer())
    bytes_per_slot = sum(
        t[:, 0].numel() * t.element_size() for t in tensors(pool).values()
    )
    assert allocated == host.size * bytes_per_slot
    assert budget < allocated <= budget + bytes_per_slot
    assert allocated == host.size * host.size_per_token


def test_capacity_is_bounded_across_reuse_and_reset(device, host_factory):
    pool = make_pool(device)
    host = host_factory(pool)
    buffer_ids = [x.data_ptr() for x in host.get_hybrid_pool_buffer()]
    all_rows = host.alloc(host.size)
    assert host.alloc(1) is None
    src, dst = torch.tensor([1, 2], device=device), torch.tensor([4, 5], device=device)
    for epoch in range(3):
        rows = all_rows[:2]
        for tensor in tensors(pool).values():
            tensor[:, src] = epoch + 17
        host.backup_from_device_all_layer(pool, rows, src)
        sync(device)
        host.free(rows)
        reused = host.alloc(2)
        assert torch.equal(reused, rows)
        for tensor in tensors(pool).values():
            tensor[:, src] = epoch + 77
        expected = snapshot(pool, src)
        host.backup_from_device_all_layer(pool, reused, src)
        sync(device)
        poison(pool, dst)
        for layer in range(pool.num_mamba_layers):
            host.load_to_device_per_layer(pool, reused, dst, layer)
        sync(device)
        assert_state(pool, dst, expected)
        assert [x.data_ptr() for x in host.get_hybrid_pool_buffer()] == buffer_ids
    host.clear()
    assert host.available_size() == host.size
    assert len(torch.unique(host.alloc(host.size))) == host.size


def test_empty_transfers_do_not_change_state(device, host_factory):
    pool = make_pool(device)
    host = host_factory(pool)
    all_rows = torch.arange(8, device=device)
    before = snapshot(pool, all_rows)
    empty_device = torch.empty(0, dtype=torch.int64, device=device)
    empty_host = torch.empty(0, dtype=torch.int64)
    host.backup_from_device_all_layer(pool, empty_host, empty_device)
    for layer in range(pool.num_mamba_layers):
        host.load_to_device_per_layer(pool, empty_host, empty_device, layer)
    sync(device)
    assert_state(pool, all_rows, before)


@pytest.mark.parametrize("kind", ["short_conv", "ngram"])
def test_disabled_companions_have_no_transfer_tensors(kind):
    pool = (
        ShortConvPool.__new__(ShortConvPool)
        if kind == "short_conv"
        else NGramPool.__new__(NGramPool)
    )
    if kind == "short_conv":
        pool.conv_state = None
    else:
        pool.context = None
    assert pool.get_slot_tensors() == ()


@pytest.mark.parametrize(
    "start,layers,ple_layer", [(0, [0, 2, 3], 1), (16, [16, 18], 17), (0, [2, 3], 0)]
)
def test_readers_wait_before_reading_restored_state(start, layers, ple_layer):
    calls = []
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.start_layer = start
    pool.mamba_map = {layer: i for i, layer in enumerate(layers)}
    pool.layer_transfer_counter = SimpleNamespace(
        wait_until=lambda n: calls.append(("wait", n))
    )
    pool.ngram_pool = SimpleNamespace(get_context=lambda _: calls.append(("ngram",)))
    pool.short_conv_pool = SimpleNamespace(
        layer_cache=lambda _: calls.append(("conv",))
    )
    pool.get_ngram_context(torch.tensor([1]))
    pool.short_conv_layer_cache(ple_layer)
    assert calls == [
        ("wait", min(layers) - start),
        ("ngram",),
        ("wait", max(ple_layer, min(layers)) - start),
        ("conv",),
    ]


def test_no_hicache_has_no_read_barrier():
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.layer_transfer_counter = None
    pool.ngram_pool = SimpleNamespace(get_context=lambda _: "history")
    pool.short_conv_pool = SimpleNamespace(layer_cache=lambda _: "conv")
    assert pool.get_ngram_context(torch.tensor([1])) == "history"
    assert pool.short_conv_layer_cache(1) == "conv"


def test_host_memory_check_includes_companions(monkeypatch, device):
    pool = make_pool(device)
    main_bytes = sum(
        t[:, 0].numel() * t.element_size()
        for t in tensors(make_pool(device, False)).values()
    )
    available = host_module.HICACHE_HOST_MEMORY_RESERVE_BYTES + 15 * main_bytes + 1
    monkeypatch.setattr(
        host_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available),
    )
    allocator = Mock(side_effect=AssertionError("Must reject before host allocation"))
    monkeypatch.setitem(host_module.ALLOC_MEMORY_FUNCS, device.type, allocator)
    with pytest.raises(ValueError, match="Not enough host memory"):
        MambaPoolHost(pool, 2, 0, layout="page_first", pin_memory=False)
    allocator.assert_not_called()


def test_partial_allocation_releases_all_pinned_buffers(monkeypatch):
    pool = make_pool(torch.device("cpu"))
    made, released = [], []

    def allocate(dims, *, dtype, **kwargs):
        if len(made) == 3:
            raise MemoryError("companion allocation failure")
        tensor = torch.empty(dims, dtype=dtype)
        made.append(tensor)
        return tensor

    monkeypatch.setattr(host_module, "_is_cuda", True)
    monkeypatch.setattr(
        host_module, "_cuda_host_unregister", lambda t: released.append(t.data_ptr())
    )
    monkeypatch.setitem(host_module.ALLOC_MEMORY_FUNCS, "cpu", allocate)
    with pytest.raises(MemoryError, match="companion allocation"):
        MambaPoolHost(pool, 2, 0, layout="page_first", pin_memory=True)
    assert released == [t.data_ptr() for t in made]


def test_destroy_is_idempotent(monkeypatch, device, host_factory):
    host = host_factory(make_pool(device))
    expected = [t.data_ptr() for t in host.get_hybrid_pool_buffer()]
    original = host_module._cuda_host_unregister
    released = []

    def unregister(tensor):
        released.append(tensor.data_ptr())
        if device.type == "cuda":
            original(tensor)

    monkeypatch.setattr(host_module, "_is_cuda", True)
    monkeypatch.setattr(host_module, "_cuda_host_unregister", unregister)
    host.pin_memory = True
    sync(device)
    host.destroy()
    host.destroy()
    assert released == expected
    assert host.sibling_buffers == []


def test_flat_state_and_pointer_metadata_cover_companions(device, host_factory):
    pool = make_pool(device)
    host = host_factory(pool)
    rows = host.alloc(2)
    host.backup_from_device_all_layer(pool, rows, torch.tensor([1, 2], device=device))
    sync(device)
    for row in rows.tolist():
        page = host.get_data_page(row)
        assert page.numel() == host.size_per_token
        saved = page.clone()
        host.set_from_flat_data_page(row, torch.zeros_like(page))
        host.set_from_flat_data_page(row, saved)
        assert torch.equal(saved, host.get_data_page(row))
    pointers, lengths = host.get_page_buffer_meta(rows)
    assert sum(lengths) == len(rows) * host.size_per_token
    expected = [t for row in rows.tolist() for t in host._iter_page_tensors(row)]
    assert pointers == [t.data_ptr() for t in expected]
    assert lengths == [t.numel() * t.element_size() for t in expected]
    assert not host.is_stride_page_aligned()


@pytest.mark.parametrize("late", [False, True])
def test_storage_attachment_rejected_before_side_effects(
    device, host_factory, monkeypatch, late
):
    host = host_factory(make_pool(device))
    controller = HybridCacheController.__new__(HybridCacheController)
    entry = SimpleNamespace(host_pool=host)
    controller.mem_pool_host = HostPoolGroup.__new__(HostPoolGroup)
    controller.mem_pool_host.entries = [] if late else [entry]
    controller.mem_pool_host.add_entry = Mock()
    controller.enable_storage = True
    controller.storage_backend = Mock()
    base_attach = Mock(side_effect=AssertionError("Storage attachment must not start"))
    monkeypatch.setattr(
        HybridCacheController.__mro__[1], "attach_storage_backend", base_attach
    )
    with pytest.raises(NotImplementedError, match="RAM and file storage only"):
        if late:
            controller.register_host_pool_entry(entry)
        else:
            controller.attach_storage_backend("mooncake")
    base_attach.assert_not_called()
    controller.mem_pool_host.add_entry.assert_not_called()


def test_non_ple_storage_guard_is_unchanged(device, host_factory):
    HybridCacheController._check_storage_pool(host_factory(make_pool(device, False)))
    HybridCacheController._check_storage_pool(SimpleNamespace())

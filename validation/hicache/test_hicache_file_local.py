"""Disk feasibility probes against the bundled file backend; no serving changes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import threading
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from test_hicache_ple_local import assert_state, make_pool, poison, snapshot

pytest_plugins = ["test_hicache_ple_local"]


def storage(path, rank=0, cap=256_000_000, metadata=True):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="qwen-ple-file-probe",
            extra_config={
                "max_size": cap,
                "min_free_space": 0,
                "eviction_ratio": 0.9,
                "enable_metadata_cache": metadata,
            },
        ),
        file_path=str(path),
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("metadata", [False, True])
def test_file_restart_restores_ple_and_recurrent_state(
    tmp_path, device, host_factory, rank, metadata
):
    tmp_path.mkdir(exist_ok=True)
    pool = make_pool(device)
    host = host_factory(pool)
    src, dst = torch.tensor([1], device=device), torch.tensor([5], device=device)
    rows = host.alloc(1)
    expected = snapshot(pool, src)
    host.backup_from_device_all_layer(pool, rows, src)
    backend = storage(tmp_path, rank, metadata=metadata)
    assert backend._evictor.max_size_bytes == 256_000_000
    backend.register_mem_host_pool_v2(host, PoolName.MAMBA)
    transfer = PoolTransfer(PoolName.MAMBA, host_indices=rows, keys=["checkpoint"])
    assert backend.batch_set_v2([transfer]) == {PoolName.MAMBA: [True]}
    for tensor in host.get_hybrid_pool_buffer():
        tensor.fill_(0)
    poison(pool, dst)

    # Reconstruct storage metadata from disk; no in-memory backend state survives.
    backend = storage(tmp_path, rank, metadata=metadata)
    assert backend._evictor._total_bytes == host.size_per_token
    backend.register_mem_host_pool_v2(host, PoolName.MAMBA)
    assert backend.batch_get_v2([transfer]) == {PoolName.MAMBA: [True]}
    for layer in range(pool.num_mamba_layers):
        host.load_to_device_per_layer(pool, rows, dst, layer)
    assert_state(pool, dst, expected)


def test_two_rank_limits_bound_aggregate_and_survive_restart(tmp_path):
    per_rank = 1024
    ranks = [storage(tmp_path, rank, per_rank) for rank in (0, 1)]

    def fill(rank):
        for i in range(20):
            assert ranks[rank].set(
                str(i), torch.full((128,), rank + 1, dtype=torch.uint8)
            )
            assert ranks[rank]._evictor._total_bytes <= per_rank

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fill, (0, 1)))
    assert sum(p.stat().st_size for p in tmp_path.glob("*.bin")) <= 2 * per_rank
    assert not list(tmp_path.glob("*.tmp.*"))
    for rank in (0, 1):
        reopened = storage(tmp_path, rank, per_rank)
        assert reopened._evictor._total_bytes <= per_rank
        page = reopened.get("19", torch.zeros(128, dtype=torch.uint8))
        assert torch.all(page == rank + 1)


def test_rejects_value_larger_than_rank_cap(tmp_path):
    backend = storage(tmp_path, cap=64)
    assert not backend.set("oversized", torch.ones(65, dtype=torch.uint8))
    assert not list(tmp_path.glob("*.bin"))
    assert backend._evictor._total_bytes == 0


def test_missing_file_is_cache_miss(tmp_path):
    assert storage(tmp_path).get("missing", torch.zeros(32, dtype=torch.uint8)) is None


def test_short_file_is_cache_miss(tmp_path):
    backend = storage(tmp_path)
    assert backend.set("short", torch.ones(16, dtype=torch.uint8))
    assert backend.get("short", torch.zeros(32, dtype=torch.uint8)) is None


def test_disk_read_error_is_cache_miss(tmp_path, monkeypatch):
    backend = storage(tmp_path)
    assert backend.set("io-error", torch.ones(16, dtype=torch.uint8))
    target = Path(backend._get_component_path("io-error"))
    original = open

    def read_error(path, mode="r", *args, **kwargs):
        if Path(path) == target and mode == "rb":
            raise OSError(5, "injected disk read error")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", read_error)
    assert backend.get("io-error", torch.zeros(16, dtype=torch.uint8)) is None


def test_failed_write_rolls_back_reservation(tmp_path, monkeypatch):
    backend = storage(tmp_path, cap=1024)

    def rename_error(*args):
        raise OSError(28, "injected filesystem full")

    monkeypatch.setattr("os.replace", rename_error)
    assert not backend.set("failed", torch.ones(128, dtype=torch.uint8))
    assert backend._evictor._total_bytes == 0
    assert not backend._evictor._pending_writes
    assert not list(tmp_path.iterdir())


def test_prefetch_worker_continues_after_short_read(tmp_path, device, host_factory):
    host = host_factory(make_pool(device))
    rows = host.alloc(2)
    backend = storage(tmp_path)
    assert backend.set("broken", torch.zeros(1, dtype=torch.uint8))
    expected = torch.full_like(host.get_dummy_flat_data_page(), 17)
    assert backend.set("valid", expected)
    controller = HiCacheController.__new__(HiCacheController)
    controller.storage_backend, controller.mem_pool_host = backend, host
    controller.page_size, controller.has_draft = 1, False
    controller.page_get_func = controller._generic_page_get
    controller.storage_stop_event = threading.Event()
    controller.prefetch_buffer, controller.host_mem_release_queue = Queue(), Queue()
    operations = []
    for i, key in enumerate(["broken", "valid"]):
        op = PrefetchOperation(key, [i])
        op.hash_value, op.host_indices = [key], rows[i : i + 1]
        operations.append(op)
        controller.prefetch_buffer.put(op)
    finished = threading.Event()
    increment = operations[1].increment

    def completed(n):
        result = increment(n)
        finished.set()
        return result

    operations[1].increment = completed
    errors = []

    def work():
        try:
            controller.prefetch_io_aux_func()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert finished.wait(3), (
            "Prefetch worker did not reach the next valid page",
            errors,
        )
    finally:
        controller.storage_stop_event.set()
        controller.prefetch_buffer.put(None)
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert operations[0].is_terminated() and operations[0].completed_tokens == 0
    assert operations[1].completed_tokens == 1
    assert controller.host_mem_release_queue.qsize() == 1
    assert torch.equal(host.get_data_page(rows[1].item()), expected)


@pytest.mark.parametrize("late", [False, True])
def test_file_attachment_allows_complete_checkpoint(
    tmp_path, device, host_factory, monkeypatch, late
):
    host = host_factory(make_pool(device))
    backend = storage(tmp_path)
    controller = HybridCacheController.__new__(HybridCacheController)
    anchor = PoolEntry(PoolName.KV, host, host.device_pool, lambda n: n, True)
    controller.mem_pool_host = HostPoolGroup([anchor])
    controller.storage_backend, controller.enable_storage = backend, True
    controller.extra_host_mem_release_queues = {}
    entry = PoolEntry(PoolName.MAMBA, host, host.device_pool, lambda n: n)
    if late:
        controller.register_host_pool_entry(entry)
        assert controller.mem_pool_host.entry_map[PoolName.MAMBA] is entry
    else:
        base_attach = Mock()
        monkeypatch.setattr(
            HybridCacheController.__mro__[1], "attach_storage_backend", base_attach
        )
        controller.attach_storage_backend("file", host_pools=[entry])
        base_attach.assert_called_once()
    assert backend.registered_pools[PoolName.MAMBA] is host


def test_metadata_queries_do_not_scan_the_entire_cache(tmp_path, monkeypatch):
    backend = storage(tmp_path)
    assert backend.set("prefix", torch.ones(32, dtype=torch.uint8))
    scan = Mock(
        side_effect=AssertionError("A prefix lookup must not scan a 512GB directory")
    )
    monkeypatch.setattr("os.scandir", scan)
    assert backend.batch_exists_v2(["prefix"]).kv_hit_pages == 1
    scan.assert_not_called()

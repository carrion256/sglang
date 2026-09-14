"""Compressed QSA cache layout, budgeting and required disk-page coverage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.qsa_pool_host import (
    QSAPagedHostPool,
    qsa_index_bytes_per_token,
)

from test_hicache_file_local import storage


def make_index_pool(layers=2, device="cpu", ratio=4, page_size=64):
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.page_size, pool.qsa_compress_ratio = page_size, ratio
    pool.qsa_compressed_k_buffer_pool = [
        ((torch.arange(80 * 128, device=device) % 173) + layer)
        .to(torch.bfloat16)
        .reshape(80, 1, 128)
        for layer in range(layers)
    ]
    pool.full_attention_layer_id_mapping = {4 * i + 3: i for i in range(layers)}
    pool.start_layer, pool.layer_transfer_counter = 0, None
    return pool


@pytest.mark.parametrize("layout", ["layer_first", "page_first", "page_first_direct"])
def test_flat_page_carries_every_target_and_draft_layer(layout):
    pools = (make_index_pool(), make_index_pool(1))
    host = QSAPagedHostPool(pools, 192, 64, layout, pin_memory=False)
    try:
        assert host.size_per_token == 3 * 128 * 2 // 4
        assert host.get_size_per_token() == host.size_per_token
        assert (
            sum(x.numel() * x.element_size() for x in host.get_hybrid_pool_buffer())
            == 192 * host.size_per_token
        )
        expected = torch.arange(64 * host.size_per_token).to(torch.uint8)
        host.set_from_flat_data_page(64, expected)
        assert torch.equal(host.get_data_page(64), expected)
        assert host.get_dummy_flat_data_page().shape == expected.shape
        assert host._to_page_indices(torch.arange(64, 192)).tolist() == [1, 2]
    finally:
        host.destroy()


@pytest.mark.parametrize("ratio,page_size", [(0, 64), (3, 64), (4, 1)])
def test_rejects_incomplete_compression_groups(ratio, page_size):
    with pytest.raises(ValueError):
        qsa_index_bytes_per_token((make_index_pool(ratio=ratio),), page_size)


def test_rejects_partial_copy_and_mismatched_draft():
    host = QSAPagedHostPool(
        (make_index_pool(),), 192, 64, "page_first", pin_memory=False
    )
    try:
        with pytest.raises(ValueError, match="complete KV pages"):
            host._has_transfer_indices(torch.arange(63), torch.arange(63))
    finally:
        host.destroy()
    with pytest.raises(ValueError, match="shapes must match"):
        QSAPagedHostPool(
            (make_index_pool(), make_index_pool(1, ratio=8)),
            192,
            64,
            "page_first",
            pin_memory=False,
        )


def test_index_read_waits_on_global_attention_layer():
    pool = make_index_pool()
    pool.layer_transfer_counter = Mock()
    result = pool.get_qsa_compressed_k_buffer(7)
    pool.layer_transfer_counter.wait_until.assert_called_once_with(7)
    assert result is pool.qsa_compressed_k_buffer_pool[1]


def test_file_requires_all_sparse_index_pages(tmp_path):
    host = QSAPagedHostPool(
        (make_index_pool(), make_index_pool(1)), 192, 64, "page_first", pin_memory=False
    )
    try:
        backend = storage(tmp_path)
        backend.register_mem_host_pool_v2(host, PoolName.QSA_INDEXER)
        keys = ["first", "second"]
        for key in keys:
            assert backend.set(key, torch.zeros(32, dtype=torch.uint8))
        transfer = PoolTransfer(
            PoolName.QSA_INDEXER,
            host_indices=torch.arange(128),
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
            indices_from_pool=PoolName.KV,
        )
        assert backend.batch_exists_v2(keys, [transfer]).kv_hit_pages == 0
        expected = torch.arange(64 * host.size_per_token).to(torch.uint8)
        host.set_from_flat_data_page(0, expected)
        first = PoolTransfer(
            PoolName.QSA_INDEXER, host_indices=torch.arange(64), keys=keys[:1]
        )
        assert backend.batch_set_v2([first]) == {PoolName.QSA_INDEXER: [True]}
        assert backend.batch_exists_v2(keys, [transfer]).kv_hit_pages == 1
        host.set_from_flat_data_page(64, expected.flip(0))
        assert backend.batch_set_v2([transfer]) == {PoolName.QSA_INDEXER: [True, True]}
        host.kv_buffer.zero_()
        reopened = storage(tmp_path)
        reopened.register_mem_host_pool_v2(host, PoolName.QSA_INDEXER)
        assert reopened.batch_get_v2([transfer]) == {PoolName.QSA_INDEXER: [True, True]}
        assert torch.equal(host.get_data_page(0), expected)
        assert torch.equal(host.get_data_page(64), expected.flip(0))
    finally:
        host.destroy()


def test_mamba_strategy_registers_required_index_and_rejects_missing_draft(monkeypatch):
    target, draft = make_index_pool(), make_index_pool(1)
    for pool, count in ((target, 2), (draft, 1)):
        pool.full_kv_pool = SimpleNamespace(layer_num=count)
    target.use_mla = False
    params = SimpleNamespace(
        page_size=64,
        mtp_draft_device_pools=(draft,),
        req_to_token_pool=SimpleNamespace(
            mamba_map={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}, mamba_pool=object()
        ),
    )
    group, controller = Mock(), Mock()
    build = Mock(return_value=(group, controller))
    monkeypatch.setattr(assembler, "build_hybrid_mamba_stack", build)
    result = assembler._MambaStrategy().build(
        cache=Mock(),
        kvcache=target,
        params=params,
        server_args=Mock(),
        load_cache_event=Mock(),
    )
    assert build.call_args.kwargs["qsa_device_pools"] == (target, draft)
    assert len(result.sidecars) == 1
    assert result.sidecars[0].pool_name == PoolName.QSA_INDEXER
    assert result.sidecars[0].hit_policy == PoolHitPolicy.ALL_PAGES
    assert result.sidecars[0].indices_from_pool == PoolName.KV
    params.mtp_draft_device_pools = (SimpleNamespace(),)
    with pytest.raises(ValueError, match="compressed QSA draft"):
        assembler._MambaStrategy().build(
            cache=Mock(),
            kvcache=target,
            params=params,
            server_args=Mock(),
            load_cache_event=Mock(),
        )


@pytest.mark.parametrize("with_draft", [False, True])
def test_fixed_host_budget_includes_index_and_draft(monkeypatch, with_draft):
    target, draft = make_index_pool(), make_index_pool(1)
    kv = SimpleNamespace(
        size=256,
        page_size=64,
        layer_num=2,
        get_kv_size_bytes=lambda: (320 * 2 * 256, 320 * 2 * 256),
    )
    draft.full_kv_pool = SimpleNamespace(
        size=256,
        page_size=64,
        layer_num=1,
        get_kv_size_bytes=lambda: (320 * 256, 320 * 256),
    )
    state = SimpleNamespace(get_kv_size_bytes=lambda: 320 * 1024)
    args = SimpleNamespace(
        hicache_size=0.01,
        hicache_ratio=2,
        hicache_mem_layout="page_first",
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
    )
    params = SimpleNamespace(
        mtp_draft_device_pools=(draft,) if with_draft else (),
        req_to_token_pool=SimpleNamespace(mamba_allocator=Mock()),
        page_size=64,
        token_to_kv_pool_allocator=Mock(),
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    built = {}

    def kv_host(**kwargs):
        built["kv_budget"] = kwargs["host_size"] * 1e9
        per_token = 1024 + (512 if with_draft else 0)
        size = (int(built["kv_budget"] // per_token) // 64 + 1) * 64
        return SimpleNamespace(
            size=size,
            logical_size=size,
            page_size=64,
            layout="page_first",
            device="cpu",
            size_per_token=per_token,
            can_use_write_back_jit=False,
        )

    def state_host(*args, **kwargs):
        built["state_budget"] = args[2] * 1e9
        return SimpleNamespace(can_use_write_back_jit=False)

    monkeypatch.setattr(assembler, "build_kv_host_pool", kv_host)
    monkeypatch.setattr(assembler, "MambaPoolHost", state_host)
    monkeypatch.setattr(assembler, "HybridCacheController", Mock())
    monkeypatch.setattr(assembler, "_get_allocator_type", lambda args: "default")
    from sglang.srt.mem_cache import qsa_pool_host as host_module

    monkeypatch.setattr(
        host_module,
        "QSAPagedHostPool",
        lambda *args, **kwargs: QSAPagedHostPool(*args, pin_memory=False, **kwargs),
    )
    group, _ = assembler.build_hybrid_mamba_stack(
        params=params,
        server_args=args,
        kv_pool=kv,
        mamba_pool=state,
        full_layer_mapping={3: 0, 7: 1},
        mamba_layer_mapping={i: i for i in (0, 1, 2, 4, 5, 6)},
        load_cache_event=Mock(),
        storage_backend=None,
        use_mla=False,
        qsa_device_pools=(target, draft) if with_draft else (target,),
    )
    index = group.get_pool(PoolName.QSA_INDEXER)
    try:
        anchor = group.get_pool(PoolName.KV)
        actual = (
            anchor.size * (anchor.size_per_token + index.size_per_token)
            + built["state_budget"]
        )
        assert (
            10_000_000
            <= actual
            <= 10_000_000 + 64 * (anchor.size_per_token + index.size_per_token)
        )
        entry = group.entry_map[PoolName.QSA_INDEXER]
        assert entry.layer_mapper(3) == 0 and entry.layer_mapper(7) == 1
        assert entry.layer_mapper(8) == (2 if with_draft else None)
    finally:
        index.destroy()

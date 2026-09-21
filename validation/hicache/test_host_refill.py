"""Real CPU cache paths; only GPU DMA is simulated by the shared fixture."""
from array import array
import time

import pytest
import torch

from test_publication_lifecycle import harness, drain, FULL, MAMBA
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.hicache_storage import PoolName


def key(tokens):
    return RadixKey(array('q', tokens))


def reclaimed(cache, parent, *, keep_full=False):
    cache.evict(EvictParams(num_tokens=0, mamba_num=1))
    drain(cache)
    assert parent.mamba_storage_success_generation == 1
    assert cache.evict_host(1, MAMBA) == 1
    if not keep_full:
        assert cache.evict_host(4, FULL) == 4
    assert parent.component_data[FULL].value is not None
    assert parent.component_data[MAMBA].value is None
    assert parent.component_data[MAMBA].host_value is None


def complete(cache, rid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        cache.check_hicache_events()
        if cache.check_prefetch_progress(rid):
            return
        time.sleep(.01)
    raise AssertionError('prefetch did not finish')


@pytest.mark.parametrize('keep_full', [False, True])
def test_real_disk_restore_repairs_existing_path(harness, keep_full):
    cache, parent, _ = harness
    reclaimed(cache, parent, keep_full=keep_full)
    before = parent.component_data[FULL].host_value
    pool = cache.cache_controller.mem_pool_host
    mamba_pool = pool.entry_map[PoolName.MAMBA].host_pool
    free_full, free_mamba = pool.available_size(), mamba_pool.available_size()
    cache.prefetch_from_storage('restore', cache.root_node.id, array('q', [1,2,3,4]))
    complete(cache, 'restore')
    assert parent.component_data[FULL].host_value is not None
    assert parent.component_data[MAMBA].host_value is not None
    assert cache.pop_prefetch_loaded_tokens('restore') == (0 if keep_full else 4)
    if keep_full:
        assert torch.equal(before, parent.component_data[FULL].host_value)
    match = cache.match_prefix(MatchPrefixParams(key=key([1,2,3,4])))
    assert len(match.device_indices) + match.host_hit_length == 4
    cache.check_hicache_events()
    assert pool.available_size() == free_full - (0 if keep_full else 4)
    assert mamba_pool.available_size() == free_mamba - 1
    # Repeated completed IO must release duplicates exactly once.
    cache.prefetch_from_storage('duplicate', cache.root_node.id, array('q', [1,2,3,4]))
    complete(cache, 'duplicate')
    cache.check_hicache_events()
    assert cache.pop_prefetch_loaded_tokens('duplicate') == 0
    assert pool.available_size() == free_full - (0 if keep_full else 4)
    assert mamba_pool.available_size() == free_mamba - 1
    cache.sanity_check()


@pytest.mark.parametrize('resident', [(False,False), (False,True), (True,False), (True,True)])
@pytest.mark.parametrize('length', [2,4,6,8,10])
def test_sparse_spans_splits_and_suffix(harness, resident, length):
    cache, parent, child = harness
    pool = cache.cache_controller.mem_pool_host
    nodes = [parent, child]
    originals = {}
    for node, present in zip(nodes, resident):
        if present:
            value = pool.alloc(4)
            node.component_data[FULL].host_value = value.clone()
            cache.tree_core._update_evictable_leaf_sets(node)
            cache.tree_core._update_evictable_leaf_sets(node.parent)
            cache.tree_core._update_duplicate_tracking(node)
            originals[node.id] = value.clone()
    incoming = pool.alloc(length)
    hashes = [str(i) for i in range(length)]
    result = cache.tree_core.insert_host(cache.root_node.id, key(range(1,length+1)),
        incoming, hashes, repair_missing=True)
    assert result.prefix_len == min(8,length)
    retained = sum(min(4,max(0,length-i*4)) for i,present in enumerate(resident) if not present)
    retained += max(0,length-8)
    assert result.retained_host_tokens == retained
    released = torch.cat(result.duplicate_host_indices) if result.duplicate_host_indices else incoming[:0]
    assert len(released) + retained == length
    assert result.inserted_host_node is not None
    current = cache.tree_core.node_by_id(result.inserted_host_node)
    path = []
    while current is not cache.root_node:
        path.append(current.component_data[FULL].host_value)
        assert path[-1] is not None
        current = current.parent
    retained_indices = torch.cat(list(reversed(path)))
    assert len(retained_indices) == length
    assert not set(retained_indices.tolist()) & set(released.tolist())
    assert set(incoming.tolist()) - set(released.tolist()) <= set(retained_indices.tolist())
    # QSA and KV payloads use the same host indices. Index adoption must not remap.
    payload = torch.arange(pool.size * 2).reshape(pool.size,2)
    adopted = incoming[~torch.isin(incoming,released)]
    assert torch.equal(payload[adopted], payload[retained_indices[torch.isin(retained_indices,adopted)]])
    for duplicate in result.duplicate_host_indices:
        pool.free(duplicate)
    cache.sanity_check()


def test_legacy_default_retains_original_contract(harness):
    cache, parent, _ = harness
    pool = cache.cache_controller.mem_pool_host
    incoming = pool.alloc(4)
    result = cache.tree_core.insert_host(cache.root_node.id, key([1,2,3,4]), incoming, ['x']*4)
    assert result.retained_host_tokens is None
    assert result.prefix_len == 4
    assert result.inserted_host_node is None
    pool.free(incoming)


def test_disagreement_releases_without_adoption(harness, monkeypatch):
    import sglang.srt.mem_cache.unified_radix_cache as module
    cache, parent, _ = harness
    reclaimed(cache, parent)
    pool = cache.cache_controller.mem_pool_host
    mamba_pool = pool.entry_map[PoolName.MAMBA].host_pool
    free_full, free_mamba = pool.available_size(), mamba_pool.available_size()
    cache.prefetch_from_storage('disagree', cache.root_node.id, array('q', [1,2,3,4]))
    monkeypatch.setattr(module, 'agree', lambda *args, **kwargs: False)
    complete(cache, 'disagree')
    assert parent.component_data[FULL].host_value is None
    assert parent.component_data[MAMBA].host_value is None
    assert 'disagree' not in cache.ongoing_prefetch
    assert cache.cache_controller.prefetch_tokens_occupied == 0
    assert cache.pop_prefetch_loaded_tokens('disagree') == 0
    cache.check_hicache_events()
    assert pool.available_size() == free_full
    assert mamba_pool.available_size() == free_mamba
    cache.sanity_check()


def test_write_through_keeps_legacy_completion(harness):
    cache, parent, _ = harness
    reclaimed(cache, parent)
    cache.is_write_back = False
    cache.tree_core.is_write_back = False
    cache.prefetch_from_storage('legacy', cache.root_node.id, array('q', [1,2,3,4]))
    complete(cache, 'legacy')
    assert parent.component_data[FULL].host_value is None
    assert parent.component_data[MAMBA].host_value is None
    assert cache.pop_prefetch_loaded_tokens('legacy') == 0


def test_pending_anchor_is_pinned_and_cancelled_prefetch_is_cleaned(harness):
    cache, parent, _ = harness
    reclaimed(cache, parent, keep_full=True)
    before = parent.component_data[FULL].host_lock_ref
    cache.prefetch_from_storage('cancel', parent.id, array('q', [5,6,7,8]), parent.hash_value[-1])
    assert parent.component_data[FULL].host_lock_ref > before
    cache.terminate_prefetch('cancel')
    complete(cache, 'cancel')
    assert parent.component_data[FULL].host_lock_ref == before
    assert 'cancel' not in cache.ongoing_prefetch
    assert cache.cache_controller.prefetch_tokens_occupied == 0
    cache.sanity_check()


def identity_worker(rank, rendezvous):
    from datetime import timedelta
    from types import SimpleNamespace as NS
    from sglang.srt.mem_cache.checkpoint_coordination import agree
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        cache = NS(tp_world_size=2)
        cache._all_reduce_attn_groups = lambda t, op: torch.distributed.all_reduce(t, op=op)
        child = NS(id=2, key=key([1,2,3,4]), children={},
            component_data=[NS(value=None,host_value=None) for _ in range(8)],
            write_through_pending_id=None, load_back_pending_id=None)
        root = NS(id=1, children={child.key.child_key(1):child})
        tree = NS(node_by_id=lambda _: root, page_size=1, components_by_type={FULL:None,MAMBA:None})
        def plan():
            return UnifiedTreeCore.host_refill_identity(tree, 1, key([1,2,3,4]))
        assert agree(cache, plan(), optional=True)
        child.component_data[FULL].host_value = torch.tensor([rank+10])
        assert agree(cache, plan(), optional=True)  # Allocator addresses are rank-local.
        if rank:
            child.component_data[FULL].host_value = None
        assert not agree(cache, plan(), optional=True)
    finally:
        torch.distributed.destroy_process_group()


def test_real_two_rank_ownership_agreement(tmp_path):
    torch.multiprocessing.spawn(identity_worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)

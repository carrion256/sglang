"""Read-only source investigation; synthetic CPU data, never production cache."""
import importlib.util
import sys
import unittest
from array import array
from types import SimpleNamespace as NS
import time

import logging
import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.hicache_storage import PoolName

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


@pytest.fixture
def harness(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('trace_fixture', '/sgl-workspace/sglang/test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py')
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture, 'get_device', lambda: 'cpu')
    pool_class = fixture.HybridReqToTokenPool
    def cpu_pool(**kw):
        kw['speculative_num_draft_tokens'] = None
        return pool_class(**kw)
    monkeypatch.setattr(fixture, 'HybridReqToTokenPool', cpu_pool)
    cfg = fixture.CacheConfig(components=(FULL, MAMBA), num_layers=2,
        full_attention_layer_ids=(0,), kv_size=32, max_context_len=32,
        max_num_reqs=2, mamba_cache_size=8)
    cache, allocator, req_pool = fixture.build_fixture(cfg)
    class Suite(fixture.UnifiedRadixCacheSuite, unittest.TestCase):
        pass
    suite = Suite()
    suite.cfg = cfg
    try:
        suite._init_hicache(cache, write_policy='write_back', storage_backend='file',
            storage_dir=str(tmp_path), prefetch_threshold=1)
        # CUDA transfer kernels have no CPU implementation. Keep the production
        # allocator, queue, commit, pin and acknowledgement paths; simulate DMA only.
        submissions = []
        def submit(transfers):
            submissions.append([(type(t.host_pool).__name__, len(t.host_indices)) for t in transfers])
            event = NS(synchronize=lambda: None, query=lambda: True)
            return NS(start_event=event, finish_event=event, timing_enabled=False)
        monkeypatch.setattr(cache.cache_controller.l2_transfer_engine,
                            'submit_device_to_host', submit)
        cache.trace_submissions = submissions
        for tokens in ([1,2,3,4], [1,2,3,4,5,6,7,8]):
            cache.insert(InsertParams(key=RadixKey(array('q', tokens)),
                value=allocator.alloc(len(tokens)), mamba_value=req_pool.mamba_allocator.alloc(1)))
        parent = next(iter(cache.root_node.children.values()))
        child = next(iter(parent.children.values()))
        yield cache, parent, child
    finally:
        suite.doCleanups()


def test_real_hicache_cpu(harness):
    cache, parent, child = harness
    result = cache.evict(EvictParams(num_tokens=0, mamba_num=1))
    assert result.mamba_num_evicted == 1
    assert parent.component_data[MAMBA].value is None
    assert parent.component_data[MAMBA].host_value is not None
    cache.sanity_check()


def drain(cache):
    deadline = time.monotonic() + 5
    while cache.ongoing_backup and time.monotonic() < deadline:
        cache.check_hicache_events()
        time.sleep(.01)
    assert not cache.ongoing_backup


@pytest.mark.parametrize('remove_mamba,remove_kv', [(False,False),(False,True),(True,False),(True,True)])
def test_host_reclamation_then_kv_rebackup(harness, caplog, remove_mamba, remove_kv):
    caplog.set_level(logging.DEBUG)
    cache, parent, child = harness
    cache.evict(EvictParams(num_tokens=0, mamba_num=1))
    drain(cache)
    assert parent.mamba_storage_success_generation == 1
    first_spec = cache.tree_core.build_storage_backup_spec(parent.id, False)
    mamba_transfer = first_spec.comp_xfers[MAMBA][0]
    backend = cache.cache_controller.storage_backend
    assert backend.batch_exists_v2(parent.hash_value, [mamba_transfer]).kv_hit_pages == 4
    print('first backup:', cache.trace_submissions, 'disk generation:', parent.mamba_storage_success_generation)
    if remove_mamba:
        assert cache.evict_host(1, MAMBA) == 1
        assert parent.component_data[MAMBA].host_value is None
    assert parent.component_data[MAMBA].value is None
    assert parent.mamba_checkpoint_required
    if remove_kv:
        assert cache.evict_host(4, FULL) == 4
        assert parent.component_data[FULL].host_value is None
    assert parent.component_data[FULL].value is not None
    cache.sanity_check()
    # Evict the actual leaf, then its parent through the production driver.
    cache.evict(EvictParams(num_tokens=8, mamba_num=0))
    drain(cache)
    fails = remove_mamba and remove_kv
    assert ('reason=missing_mamba' in caplog.text) == fails
    assert parent.mamba_storage_failures == ({2} if fails else set())
    assert parent.mamba_storage_success_generation == (2 if remove_kv and not remove_mamba else 1)
    assert backend.batch_exists_v2(parent.hash_value, [mamba_transfer]).kv_hit_pages == 4
    print('all transfers:', cache.trace_submissions,
          'prior disk checkpoint still exists:', True,
          'failed generation:', parent.mamba_storage_failures)
    cache.sanity_check()


def test_pending_disk_publication_pins_both_host_pools(harness):
    cache, parent, child = harness
    cache.evict(EvictParams(num_tokens=0, mamba_num=1))
    assert cache.ongoing_backup
    assert parent.component_data[FULL].host_lock_ref > 0
    assert parent.component_data[MAMBA].host_lock_ref > 0
    assert cache.evict_host(1, MAMBA) == 0
    assert cache.evict_host(4, FULL) == 0
    drain(cache)
    assert parent.component_data[FULL].host_lock_ref == 0
    assert parent.component_data[MAMBA].host_lock_ref == 0
    cache.sanity_check()


def test_allocator_recovery_reaches_missing_checkpoint(harness, caplog):
    caplog.set_level(logging.DEBUG)
    cache, parent, child = harness
    cache.evict(EvictParams(num_tokens=0, mamba_num=1))
    drain(cache)
    saved_spec = cache.tree_core.build_storage_backup_spec(parent.id, False)
    saved_mamba = saved_spec.comp_xfers[MAMBA][0]
    controller = cache.cache_controller
    full_pool = controller.mem_pool_host
    mamba_pool = full_pool.entry_map[PoolName.MAMBA].host_pool
    # Consume free host slots through their real allocators. The next backup
    # must invoke production reservation recovery and select its own victims.
    held_full = full_pool.alloc(full_pool.available_size())
    held_mamba = mamba_pool.alloc(mamba_pool.available_size())
    assert held_full is not None and held_mamba is not None
    assert cache._execute_and_commit_kv_backup(
        cache.tree_core._build_backup_kv_action(child, True), write_back=True)
    cache.writing_check(write_back=True)
    drain(cache)
    assert parent.component_data[MAMBA].value is None
    assert parent.component_data[MAMBA].host_value is None
    assert parent.component_data[FULL].host_value is None
    assert parent.component_data[FULL].value is not None
    assert 'reason=missing_mamba' not in caplog.text
    cache.sanity_check()
    # End external fixture pressure before the subsequent normal GPU eviction.
    full_pool.free(held_full)
    mamba_pool.free(held_mamba)
    cache.evict(EvictParams(num_tokens=8, mamba_num=0))
    drain(cache)
    assert 'reason=missing_mamba' in caplog.text
    assert 'publication_history=prior_success generation=2 prior_success_generation=1' in caplog.text
    assert 'device_mamba=False host_mamba=False checkpoint_required=True storage_residency=unverified' in caplog.text
    assert cache.cache_diagnostics.repeat_publications >= 1
    assert parent.mamba_storage_failures == {2}
    assert parent.mamba_storage_success_generation == 1
    assert controller.storage_backend.batch_exists_v2(parent.hash_value, [saved_mamba]).kv_hit_pages == 4
    print('allocator-recovery sequence:', cache.trace_submissions)
    cache.sanity_check()

"""CPU-only failure/ownership tests against the prepared runtime methods."""
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock
import logging
import threading

import pytest
import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, _OngoingWriteThrough
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_cache.components import ComponentType, CacheTransferPhase
from sglang.srt.mem_cache.unified_cache.components.mamba_component import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import EvictLayer
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController, StorageOperation
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult
from collections import defaultdict

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA

def node():
    n = UnifiedTreeNode((FULL, MAMBA))
    n.component_data[FULL].value = torch.tensor([1, 2])
    n.component_data[MAMBA].value = torch.tensor([3])
    n.hash_value = ['a', 'b']
    return n


@pytest.mark.parametrize('success', [True, False])
def test_preserve_waits_before_eviction_or_logs_loss(success, caplog):
    n = node()
    cache = object.__new__(UnifiedRadixCache)
    cache.cache_controller = object()
    cache.tree_core = NS(node_by_id=lambda _: n, is_write_back=True)
    events = []
    def copy(*a, **kw):
        events.append('submit')
        if success:
            n.component_data[MAMBA].host_value = torch.tensor([4])
        return success
    cache._execute_and_commit_kv_backup = copy
    cache.writing_check = lambda **kw: events.append('complete')
    cache.preserve_mamba_before_eviction(n.id)
    assert events == (['submit', 'complete'] if success else ['submit'])
    assert n.mamba_checkpoint_required and not n.mamba_backup_in_progress
    assert ('reason=checkpoint_loss' in caplog.text) == (not success)


def test_existing_pending_host_copy_is_completed():
    n = node()
    n.component_data[MAMBA].host_value = torch.tensor([5])
    n.write_through_pending_id = n.id
    c = NS(cache_controller=object(), is_write_back=True,
           tree_core=NS(node_by_id=lambda _: n), writing_check=Mock())
    UnifiedRadixCache.preserve_mamba_before_eviction(c, n.id)
    c.writing_check.assert_called_once_with(write_back=True)


@pytest.mark.parametrize('auxiliary_only', [True, False])
def test_backup_success_and_two_distinct_host_pins(auxiliary_only):
    cache = object.__new__(UnifiedRadixCache)
    events = []
    def lock(_):
        token = len([e for e in events if e[0] == 'lock'])
        events.append(('lock', token))
        return NS(to_dec_params=lambda: token)
    cache.inc_host_lock_ref = lock
    cache.dec_host_lock_ref = lambda *a: events.append(('unlock', a))
    cache.tree_core = NS(
        build_backup_spec=lambda _: (torch.tensor([] if auxiliary_only else [1]), {'m': [PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([3]))]}),
        commit_backup=lambda *a: events.append(('commit',)))
    cache._build_backup_sidecar = lambda *a: []
    cache._execute_kv_backup = lambda *a: torch.tensor([] if auxiliary_only else [2])
    cache._track_write_through_node = lambda n, d, h: events.append(('track', h))
    assert cache._execute_and_commit_kv_backup(BackupKV([1]), write_back=True) is True
    assert events == [('lock', 0), ('commit',), ('lock', 1), ('track', (0, 1))]


@pytest.mark.parametrize('failure', ['allocation', 'cuda'])
def test_preallocation_lock_released_on_failure_without_swallowing_cuda(failure, caplog):
    cache = object.__new__(UnifiedRadixCache)
    cache.inc_host_lock_ref = lambda _: NS(to_dec_params=lambda: 'old')
    cache.dec_host_lock_ref = Mock()
    cache.tree_core = NS(build_backup_spec=lambda _: (torch.tensor([1]), {}))
    cache._build_backup_sidecar = lambda *a: []
    def copy(*a):
        if failure == 'cuda':
            raise RuntimeError('synthetic copy failure')
        return None
    cache._execute_kv_backup = copy
    if failure == 'cuda':
        with pytest.raises(RuntimeError):
            cache._execute_and_commit_kv_backup(BackupKV([1]), write_back=True)
    else:
        assert cache._execute_and_commit_kv_backup(BackupKV([1]), write_back=True) is False
        assert 'backup failed phase=host_allocation' in caplog.text
    if failure == 'cuda':
        cache.dec_host_lock_ref.assert_not_called()
    else:
        cache.dec_host_lock_ref.assert_called_once_with(1, 'old')


def test_ack_publishes_before_releasing_host_pins():
    cache = object.__new__(UnifiedRadixCache)
    events = []
    cache.ongoing_write_through = {1: _OngoingWriteThrough(1, 'device', [1, 2], ('before', 'after'))}
    cache.tree_core = NS(finish_write_through=lambda *a: events.append('finish'))
    cache.enable_storage = True
    cache.write_backup_storage = lambda n: events.append(('disk-pin', n))
    cache.dec_host_lock_ref = lambda n, p: events.append(('host-unpin', p))
    cache.dec_lock_ref = lambda *a: events.append('device-unpin')
    cache._finish_write_through_ack(1)
    assert events == ['finish', ('disk-pin', 1), ('disk-pin', 2),
                      ('host-unpin', 'after'), ('host-unpin', 'before'), 'device-unpin']


@pytest.mark.parametrize('required,live,host,accepted', [
    (False, False, False, True), (True, False, False, False),
    (False, True, False, False), (True, True, True, True),
])
def test_publication_guard_preserves_valid_intermediate_kv(required, live, host, accepted, caplog):
    n = node()
    n.key = NS(token_ids=[1, 2])
    n.mamba_checkpoint_required = required
    n.component_data[FULL].host_value = torch.tensor([5, 6])
    n.component_data[MAMBA].value = torch.tensor([3]) if live else None
    n.component_data[MAMBA].host_value = torch.tensor([4]) if host else None
    core = NS(node_by_id=lambda _: n, components=[])
    result = UnifiedTreeCore.build_storage_backup_spec(core, n.id, False)
    assert (result is not None) == accepted
    assert ('reason=missing_mamba' in caplog.text) == (not accepted)


def test_publication_rejects_missing_hash(caplog):
    n = node()
    n.hash_value = None
    n.component_data[FULL].host_value = torch.tensor([5, 6])
    core = NS(node_by_id=lambda _: n, components=[])
    assert UnifiedTreeCore.build_storage_backup_spec(core, n.id, False) is None
    assert 'reason=missing_hash' in caplog.text


def operation():
    return StorageOperation(torch.tensor([1, 2]), [7, 8], hash_value=['a', 'b'],
        prefix_keys=['ancestor'], pool_transfers=[PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.tensor([3]), keys=['b'])])


@pytest.mark.parametrize('result', [[], [False], [True, True], None])
def test_sidecar_failure_blocks_primary_publication(result, monkeypatch):
    c = object.__new__(HybridCacheController)
    c.should_backup = lambda _: True
    c._resolve_sidecar_derived_pool_transfers = lambda _: None
    c.storage_backend = NS(batch_set_v2=lambda _: {PoolName.MAMBA: result})
    c.backup_skip = False
    c.page_size = 1
    primary = Mock()
    monkeypatch.setattr(HiCacheController, '_page_backup', primary)
    op = operation()
    c._page_backup(op)
    primary.assert_not_called()
    assert op.backup_failure == 'sidecar_write'


@pytest.mark.parametrize('kind', ['transient', 'permanent', 'partial'])
def test_worker_retries_releases_and_serves_next_operation(kind, caplog):
    c = object.__new__(HybridCacheController)
    c.storage_stop_event = threading.Event()
    c.backup_queue = Queue()
    c.ack_backup_queue = Queue()
    c.page_size = 1
    first, second = operation(), operation()
    c.register_backup_publication(first)
    c.register_backup_publication(second)
    c.backup_queue.put(first)
    c.backup_queue.put(second)
    calls = []
    def write(op):
        calls.append(op.id)
        assert op.completed_tokens == 0 and op.prefix_keys == ['ancestor']
        if op is first and (kind != 'transient' or calls.count(first.id) == 1):
            op.prefix_keys.append('partial')
            op.completed_tokens = 1
            if kind == 'partial':
                op.backup_failure = 'primary_write'
                return
            raise OSError('synthetic IO failure')
        op.completed_tokens = 2
        if op is second:
            c.storage_stop_event.set()
    c._page_backup = write
    c.backup_thread_func()
    assert calls == [first.id] * (2 if kind == 'transient' else 4) + [second.id]
    assert c.ack_backup_queue.get_nowait() is first
    assert c.ack_backup_queue.get_nowait() is second
    assert 'backup recovery' in caplog.text
    assert ('backup failed phase=storage' in caplog.text) == (kind != 'transient')
    assert first.completed_tokens == (2 if kind == 'transient' else 1)


def test_component_allocation_failure_rolls_back_and_logs(caplog):
    c = object.__new__(HybridCacheController)
    freed, evictions = [], []
    good = NS(alloc=lambda n: torch.tensor([4]), free=lambda x: freed.append(x.tolist()))
    bad = NS(alloc=lambda n: None, free=Mock())
    c.mem_pool_host = NS(entry_map={
        PoolName.MAMBA: NS(host_pool=good, host_evict_fn=None),
        PoolName.QSA_INDEXER: NS(host_pool=bad, host_evict_fn=lambda n: evictions.append(n))})
    transfers = [PoolTransfer(name=name, device_indices=torch.tensor([1]))
                 for name in (PoolName.MAMBA, PoolName.QSA_INDEXER)]
    assert c._resolve_pool_transfers_allocation(transfers, True) is None
    assert freed == [[4]] and evictions == [1]
    assert transfers[0].host_indices is None
    assert 'transfer recovery' in caplog.text and 'transfer failed' in caplog.text


@pytest.mark.parametrize('derived', [False, True])
def test_unknown_pool_fails_before_submission(caplog, derived):
    c = object.__new__(HybridCacheController)
    c.mem_pool_host = NS(entry_map={})
    assert c._resolve_pool_transfers_allocation(
        [PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([1]),
                      indices_from_pool=PoolName.KV if derived else None)], True) is None
    assert 'unregistered_pool' in caplog.text


@pytest.mark.parametrize('layer,expected', [(EvictLayer.DEVICE, True), (EvictLayer.HOST, False), (EvictLayer.ALL, False)])
@pytest.mark.parametrize('component', [FULL, MAMBA])
def test_preservation_hook_never_starts_backup_during_deletion(layer, expected, component):
    events = []
    lru = NS(in_list=lambda _: False)
    comp = NS(component_type=component, evict_component=lambda *a, **kw: (events.append('free') or (1, 0)))
    mamba = NS(cache=NS(preserve_mamba_before_eviction=lambda _: events.append('preserve')))
    core = NS(is_write_back=True, components_by_type={MAMBA: mamba},
              lru_lists={component: lru}, host_lru_lists={component: lru})
    UnifiedTreeCore._evict_component_and_detach_lru(core, node(), comp, {}, {}, layer)
    assert events == (['preserve', 'free'] if expected else ['free'])


def test_mamba_split_and_new_host_copy_release_exact_locks():
    n, prefix, root = node(), node(), node()
    n.parent = prefix
    prefix.parent = root
    component = object.__new__(MambaComponent)
    component.tree_core = NS(root_node=root, host_lru_lists={MAMBA: NS(in_list=lambda _: False)})
    before = component.acquire_component_lock(n, IncLockRefResult(), lock_host=True).to_dec_params()
    assert n.component_data[MAMBA].host_lock_ref == 0
    n.component_data[MAMBA].host_value = torch.tensor([6])
    after = component.acquire_component_lock(n, IncLockRefResult(), lock_host=True).to_dec_params()
    assert n.component_data[MAMBA].host_lock_ref == 1
    n.mamba_checkpoint_required = True
    component.redistribute_on_node_split(prefix, n)
    assert prefix.component_data[MAMBA].host_value is None
    assert not prefix.mamba_checkpoint_required and n.mamba_checkpoint_required
    component.release_component_lock(n, after, lock_host=True)
    component.release_component_lock(n, before, lock_host=True)
    assert n.component_data[MAMBA].host_lock_ref == 0


def test_incremental_backup_pins_existing_full_before_allocation():
    n = node()
    n.component_data[FULL].host_value = torch.tensor([5, 6])
    n.component_data[FULL].host_lock_ref = 0
    cache = object.__new__(UnifiedRadixCache)
    tokens = []
    def lock(_):
        n.component_data[FULL].host_lock_ref += 1
        token = object()
        tokens.append(token)
        return NS(to_dec_params=lambda: token)
    cache.inc_host_lock_ref = lock
    cache.dec_host_lock_ref = Mock()
    cache.tree_core = NS(build_backup_spec=lambda _: (torch.tensor([]), {'m': [PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([3]))]}),
                        commit_backup=Mock())
    cache._build_backup_sidecar = lambda *a: []
    def allocate(*a):
        assert not UnifiedTreeCore._can_reclaim_full_host_duplicate(NS(root_node=None), n)
        return torch.tensor([])
    cache._execute_kv_backup = allocate
    cache._track_write_through_node = Mock()
    assert cache._execute_and_commit_kv_backup(BackupKV([n.id]), write_back=True)
    assert len(tokens) == 2


@pytest.mark.parametrize('skip', [True, False])
def test_complete_sidecar_permits_primary_and_correct_accounting(skip, monkeypatch):
    c = object.__new__(HybridCacheController)
    c.should_backup = lambda _: True
    c._resolve_sidecar_derived_pool_transfers = lambda _: None
    c.storage_backend = NS(batch_set_v2=lambda _: {PoolName.MAMBA: [True]})
    c.backup_skip = skip
    c.page_size = 1
    def primary(op):
        op.completed_tokens = 2
    primary_mock = Mock(side_effect=primary)
    monkeypatch.setattr(HiCacheController, '_page_backup', primary_mock)
    op = operation()
    c._page_backup(op)
    assert op.completed_tokens == 2
    assert primary_mock.call_count == (0 if skip else 1)


def test_component_allocation_recovers_once(caplog):
    c = object.__new__(HybridCacheController)
    allocations = iter([None, torch.tensor([8])])
    evict = Mock()
    pool = NS(alloc=lambda _: next(allocations), free=Mock())
    c.mem_pool_host = NS(entry_map={PoolName.MAMBA: NS(host_pool=pool, host_evict_fn=evict)})
    transfers = [PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([1]))]
    assert c._resolve_pool_transfers_allocation(transfers, True) == transfers
    evict.assert_called_once_with(1)
    assert 'transfer recovery' in caplog.text and 'transfer failed' not in caplog.text


@pytest.mark.parametrize('case', ['internal', 'internal_failure', 'partial_leaf', 'partial_leaf_failure', 'path_cap', 'split_pending'])
def test_real_cpu_tree_eviction_preserves_checkpoint_or_continues(monkeypatch, caplog, case, configure=None):
    import importlib.util
    import sys
    from array import array
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    name = 'checkpoint_cpu_fixture'
    spec = importlib.util.spec_from_file_location(name,
        '/sgl-workspace/sglang/test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py')
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[name] = fixture
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture, 'get_device', lambda: 'cpu')
    # Upstream's speculative intermediate allocation hardcodes CUDA. This
    # fixture tests cache bookkeeping only, so omit those unused model buffers.
    pool_class = fixture.HybridReqToTokenPool
    def cpu_pool(**kw):
        kw['speculative_num_draft_tokens'] = None
        return pool_class(**kw)
    monkeypatch.setattr(fixture, 'HybridReqToTokenPool', cpu_pool)
    cfg = fixture.CacheConfig(components=(FULL, MAMBA), num_layers=2,
        full_attention_layer_ids=(0,), kv_size=32, max_context_len=32,
        max_num_reqs=2, mamba_cache_size=8)
    cache, allocator, req_pool = fixture.build_fixture(cfg)
    cache.is_write_back = True
    for tokens in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]):
        cache.insert(InsertParams(key=RadixKey(array('q', tokens)),
            value=allocator.alloc(len(tokens)), mamba_value=req_pool.mamba_allocator.alloc(1)))
    root = cache.tree_core.root_node
    internal = next(iter(root.children.values()))
    assert internal.children
    target = next(iter(internal.children.values())) if case.startswith('partial_leaf') else internal
    # Storage was disabled during fixture construction; production assigns a
    # chained hash to every page before publishing nodes.
    internal.hash_value = ['a', 'b', 'c', 'd']
    next(iter(internal.children.values())).hash_value = ['e', 'f', 'g', 'h']
    original = target.component_data[FULL].value.clone()
    if case.startswith('partial_leaf'):
        target.component_data[FULL].host_value = torch.tensor([20, 21, 22, 23])
    events = []
    controller = NS(write_policy='write_back', ack_write_queue=[],
                    register_backup_publication=lambda op:None,
                    mem_pool_host=NS(available_size=lambda: 100))
    def write(values, node_id, extra_pools):
        assert node_id == target.id
        if case.startswith('partial_leaf'):
            assert values.numel() == 0
        if case.endswith('failure'):
            return None
        for transfer in extra_pools or []:
            transfer.host_indices = torch.tensor([9])
        finish = NS(synchronize=lambda: events.append('copy_complete'))
        controller.ack_write_queue.append(NS(finish_event=finish, node_ids=[node_id]))
        return torch.arange(len(values), dtype=torch.int64)
    controller.write = write
    cache.cache_controller = controller
    cache.enable_storage = False
    cache._build_backup_sidecar = lambda *a: []
    cache._log_write_ack_metrics = lambda _: None
    if configure is not None:
        configure(cache, controller, target, events, case)
    fragments = []
    if case == 'split_pending':
        cache.enable_storage = True
        cache.hicache_storage_pass_prefix_keys = False
        published = []
        def storage_write(*args, **kw):
            published.append((args, kw))
            return len(published)
        controller.write_storage = storage_write
        track = cache._track_write_through_node
        def split_after_submit(*args):
            track(*args)
            prefix, action = cache.tree_core._split_node(target.key, target, 2)
            fragments.append(prefix)
            if action is not None:
                cache._apply_cache_action(action)
        cache._track_write_through_node = split_after_submit
    if case.startswith('partial_leaf'):
        result = cache.tree_core.evict_device_leaf(target.id, True)
        cache._free_values(result.device_frees, result.host_frees)
        assert target.component_data[FULL].value is None
    elif case == 'path_cap':
        component = cache.components[MAMBA]
        component.mamba_max_states_per_path = 1
        from sglang.srt.mem_cache.unified_cache.cache_action import MambaEvictExcessPathStates
        component.apply_component_action(MambaEvictExcessPathStates(next(iter(internal.children.values())).id))
        assert torch.equal(target.component_data[FULL].value, original)
    else:
        result = cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        assert result.mamba_num_evicted == 1
        if case == 'split_pending':
            assert torch.equal(torch.cat([fragments[0].component_data[FULL].value,
                                          target.component_data[FULL].value]), original)
        else:
            assert torch.equal(target.component_data[FULL].value, original)
    assert target.component_data[MAMBA].value is None
    if case.endswith('failure'):
        assert target.component_data[MAMBA].host_value is None
        assert 'reason=checkpoint_loss' in caplog.text
        assert events == []
    else:
        assert target.component_data[MAMBA].host_value.numel() == 1
        assert events == ['copy_complete']
    if case == 'split_pending':
        assert len(published) == 2
        assert published[0][1]['extra_pools'] is None
        assert published[1][1]['extra_pools'][0].name == PoolName.MAMBA
        for node_id, params in list(cache.ongoing_backup.values()):
            cache.dec_host_lock_ref(node_id, params)
        cache.ongoing_backup.clear()
        assert fragments[0].component_data[FULL].host_lock_ref == 0
    assert target.component_data[FULL].host_lock_ref == 0
    assert target.component_data[MAMBA].host_lock_ref == 0
    cache.sanity_check()

@pytest.mark.parametrize('pool_name', ['mamba', 'int8'])
@pytest.mark.parametrize('recovers', [False, True])
@pytest.mark.parametrize('log_level', [logging.INFO, logging.DEBUG])
def test_optional_creation_allocation_is_bounded_and_logged(pool_name, recovers, log_level, caplog):
    caplog.set_level(log_level, logger='sglang.srt.mem_cache.checkpoint_coordination')
    from sglang.srt.mem_cache.checkpoint_coordination import reserve
    pool = NS(alloc=Mock(side_effect=[None, torch.tensor([9]) if recovers else None]), free=Mock())
    c = NS(cache=NS(req_to_token_pool=NS(mamba_allocator=pool), evict=Mock()))
    result = reserve(c.cache, 'creation', [(pool_name, 1, pool.alloc, pool.free)], c.cache.evict)
    assert (result is not None) == recovers
    assert pool.alloc.call_count == 2 and c.cache.evict.call_count == 1
    assert ('checkpoint recovery' in caplog.text) == (log_level == logging.DEBUG)
    assert ('reason=exhausted' in caplog.text) == (not recovers)


@pytest.mark.parametrize('extra,int8,fail', [(True, False, 'active'), (False, False, 'active'),
    (True, True, 'checkpoint'), (True, True, 'active'), (False, True, 'checkpoint')])
def test_skipped_creation_keeps_request_state(extra, int8, fail):
    ckpt = NS(alloc=Mock(return_value=None if fail == "checkpoint" else torch.tensor([10])), free=Mock(), store_from_active=Mock()) if int8 else None
    pool = NS(donate_mamba_ping_pong_slot=Mock(), mamba_pool=NS(copy_from=Mock()),
              mamba_allocator=NS(alloc=Mock(return_value=None), free=Mock()))
    c = NS(cache=NS(enable_mamba_extra_buffer=extra, req_to_token_pool=pool, evict=Mock()), int8_ckpt_pool=ckpt,
           _alloc_mamba_slot=Mock(return_value=None),
           _alloc_int8_ckpt_slot=Mock(return_value=None if fail == 'checkpoint' else torch.tensor([10])),
           _commit_int8_checkpoint=Mock(return_value=None))
    req = NS(mamba_last_track_seqlen=64, mamba_pool_idx=torch.tensor(2))
    params = NS(mamba_value=None)
    assert MambaComponent.prepare_for_caching_req(c, req, params, 64, False) == 0
    assert params.mamba_value is None and req.mamba_last_track_seqlen == 64
    pool.donate_mamba_ping_pong_slot.assert_not_called()
    pool.mamba_pool.copy_from.assert_not_called()
    MambaComponent.cleanup_after_caching_req(c, req, False, insert_params=params)
    assert req.mamba_last_track_seqlen == 64
    if extra and int8 and fail == 'active':
        assert ckpt.free.call_count == 2


def test_preparation_exception_releases_pin_and_continues(caplog):
    cache = NS(inc_host_lock_ref=lambda _: NS(to_dec_params=lambda: 'pin'),
               tree_core=NS(build_backup_spec=Mock(side_effect=MemoryError())),
               dec_host_lock_ref=Mock())
    assert not UnifiedRadixCache._execute_and_commit_kv_backup(cache, BackupKV([1]), True)
    cache.dec_host_lock_ref.assert_called_once_with(1, 'pin')
    assert 'phase=preparation' in caplog.text and 'retain_pins' not in caplog.text


def test_allocation_exception_rolls_back_before_submission(caplog):
    first = NS(alloc=Mock(return_value=torch.tensor([7])), free=Mock())
    second = NS(alloc=Mock(side_effect=MemoryError()), free=Mock())
    anchor = NS(alloc=Mock(return_value=torch.tensor([8])), free=Mock(), entry_map={
        PoolName.MAMBA: NS(host_pool=first, host_evict_fn=None),
        PoolName.QSA_INDEXER: NS(host_pool=second, host_evict_fn=None)})
    c = object.__new__(HybridCacheController)
    c.mem_pool_host = anchor
    c.start_writing = Mock()
    c.write_queue = []
    transfers = [PoolTransfer(name=name, device_indices=torch.tensor([1])) for name in anchor.entry_map]
    assert c.write(torch.tensor([1]), extra_pools=transfers) is None
    first.free.assert_called_once()
    anchor.free.assert_called_once()
    c.start_writing.assert_not_called()
    assert c.write_queue == [] and 'phase=allocation' in caplog.text


def test_terminal_ack_records_generation_without_rank_local_requeue(caplog):
    n = node()
    n.mamba_storage_pending = {42:1, 43:2, 44:3}
    c = NS(ongoing_backup={op:(n.id, 'pin') for op in (42,43,44)},
           tree_core=NS(node_by_id=lambda _:n), dec_host_lock_ref=Mock())
    UnifiedRadixCache._finish_storage_backup(c, NS(id=43, backup_failure='sidecar_write'))
    UnifiedRadixCache._finish_storage_backup(c, NS(id=42, backup_failure=None))
    assert n.mamba_storage_failures == {2}
    UnifiedRadixCache._finish_storage_backup(c, NS(id=44, backup_failure=None))
    assert not n.mamba_storage_failures
    n.mamba_storage_pending[45] = 1
    c.ongoing_backup[45] = (n.id, 'pin')
    UnifiedRadixCache._finish_storage_backup(c, NS(id=45, backup_failure='old_failure'))
    assert not n.mamba_storage_failures and not c.ongoing_backup
    assert c.dec_host_lock_ref.call_count == 4


def test_enqueue_failure_marks_dirty_and_releases_pin(caplog):
    n = node()
    spec = NS(host_value=torch.tensor([1,2]), hash_value=['a','b'], token_ids=[1,2], prefix_keys=None, comp_xfers={})
    c = NS(enable_storage=True, tree_core=NS(node_by_id=lambda _:n, build_storage_backup_spec=lambda *a:spec),
           hicache_storage_pass_prefix_keys=False, _build_sidecar_transfers=lambda *a:[],
           inc_host_lock_ref=lambda _:NS(to_dec_params=lambda:'pin'), dec_host_lock_ref=Mock(),
           cache_controller=NS(write_storage=Mock(side_effect=MemoryError()), ack_backup_queue=Queue(),
               register_backup_publication=lambda op:None), ongoing_backup={})
    c.cache_controller.complete_backup_publication=c.cache_controller.ack_backup_queue.put
    UnifiedRadixCache.write_backup_storage(c, n.id)
    c.dec_host_lock_ref.assert_not_called()
    UnifiedRadixCache._finish_storage_backup(c, c.cache_controller.ack_backup_queue.get_nowait())
    assert n.mamba_storage_failures and not c.ongoing_backup
    c.dec_host_lock_ref.assert_called_once_with(n.id, 'pin')
    assert 'phase=storage_enqueue' in caplog.text


@pytest.mark.parametrize('remaining', [0, 1, 2])
def test_final_publication_rejects_self_eviction(monkeypatch, remaining):
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile, PoolTransferResult
    backend = object.__new__(HiCacheFile)
    backend.batch_set_v2 = lambda transfers: {PoolName.MAMBA:[True]}
    backend.batch_exists_v2 = lambda *a: PoolTransferResult(remaining, {})
    c = object.__new__(HybridCacheController)
    c.storage_backend = backend
    c.backup_skip = False
    c.page_size = 1
    monkeypatch.setattr(HiCacheController, '_page_backup', lambda self, op: setattr(op, 'completed_tokens', 2))
    op = operation()
    c._page_backup(op)
    assert getattr(op, 'backup_failure', None) == ('publication_evicted' if remaining < 2 else None)


def test_finished_int8_failure_frees_request_once_without_publishing():
    pool = NS(get_mamba_ping_pong_keep_idx=lambda req:0, free_mamba_cache=Mock())
    c = NS(cache=NS(enable_mamba_extra_buffer=True, req_to_token_pool=pool, evict=Mock()),
           int8_ckpt_pool=NS(alloc=Mock(return_value=None), free=Mock()), _commit_int8_checkpoint=lambda _:None)
    req = NS(mamba_last_track_seqlen=64, mamba_ping_pong_track_buffer=torch.tensor([3,4]))
    params = NS(mamba_value=None)
    assert MambaComponent.prepare_for_caching_req(c, req, params, 64, True) == 0
    MambaComponent.cleanup_after_caching_req(c, req, True, insert_params=params)
    pool.free_mamba_cache.assert_called_once_with(req)
    assert params.mamba_value is None


def test_optional_donation_succeeds_after_skip():
    pool = NS(donate_mamba_ping_pong_slot=Mock(return_value=torch.tensor([3])),
              mamba_allocator=NS(alloc=Mock(side_effect=[None, None, torch.tensor([9])]), free=Mock()))
    c = NS(cache=NS(enable_mamba_extra_buffer=True, req_to_token_pool=pool, evict=Mock()), int8_ckpt_pool=None,
           _alloc_mamba_slot=Mock(side_effect=[None, torch.tensor([9])]))
    req = NS(mamba_last_track_seqlen=64)
    params = NS(mamba_value=None)
    assert MambaComponent.prepare_for_caching_req(c, req, params, 64, False) == 0
    MambaComponent.cleanup_after_caching_req(c, req, False, insert_params=params)
    assert MambaComponent.prepare_for_caching_req(c, req, params, 64, False) == 64
    assert params.mamba_value.tolist() == [3]
    assert pool.donate_mamba_ping_pong_slot.call_count == 1


def test_skipped_unfinished_caching_never_reclaims_active_slots(monkeypatch):
    from sglang.srt.environ import envs
    monkeypatch.setattr(envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS, 'get', lambda:True)
    comp = NS(prepare_for_caching_req=lambda **kw:0, free_out_of_window_slots=Mock(), cleanup_after_caching_req=Mock())
    c = NS(session=NS(try_cache_unfinished_req=lambda *a, **kw:False), disable=False,
           req_to_token_pool=NS(req_to_token=torch.tensor([[1,2,3,4]])), _components_tuple=(comp,),
           tree_core=NS(is_eagle=False))
    req = NS(get_fill_ids=lambda:[5,6,7,8], req_pool_idx=0, cache_protected_len=2,
             extra_key=None, cache_salt=None, last_node=99)
    UnifiedRadixCache.cache_unfinished_req(c, req)
    comp.free_out_of_window_slots.assert_not_called()
    assert req.cache_protected_len == 2 and req.last_node == 99
    assert req.prefix_indices.tolist() == [1,2,3,4]


@pytest.mark.parametrize('layer,live,expected', [(EvictLayer.HOST, False, True),
    (EvictLayer.HOST, True, False), (EvictLayer.ALL, True, True)])
def test_last_copy_loss_logged_once_and_eviction_continues(layer, live, expected, caplog):
    n = node()
    n.component_data[MAMBA].host_value = torch.tensor([7])
    if not live:
        n.component_data[MAMBA].value = None
    n.mamba_storage_failures.add(1)
    lru = NS(in_list=lambda _:False)
    comp = NS(component_type=MAMBA, evict_component=Mock(return_value=(0,1)))
    core = NS(is_write_back=True, components_by_type={MAMBA:object()},
              lru_lists={MAMBA:lru}, host_lru_lists={MAMBA:lru})
    for _ in range(2):
        UnifiedTreeCore._evict_component_and_detach_lru(core,n,comp,{}, {},layer)
    assert comp.evict_component.call_count == 2
    assert caplog.text.count('phase=host_eviction') == int(expected)


@pytest.mark.parametrize('metadata', [False, True])
def test_real_file_lru_self_eviction_is_not_reported_success(tmp_path, monkeypatch, metadata):
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
    config = HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False, enable_storage_metrics=False,
        is_page_first_layout=True, model_name='synthetic', extra_config={
            'max_size':100, 'min_free_space':0, 'eviction_ratio':1.0, 'enable_metadata_cache':metadata})
    backend = HiCacheFile(config, file_path=str(tmp_path))
    assert backend.set('a', torch.zeros(40,dtype=torch.uint8))
    assert backend.set('b', torch.zeros(40,dtype=torch.uint8))
    # Real file writer/evictor and final lookup; synthetic tensors replace model pools.
    backend.batch_set_v2 = lambda transfers:{PoolName.MAMBA:[backend.set('c.mamba',torch.zeros(20,dtype=torch.uint8))]}
    def primary(self, op):
        for key, size in [('a',40),('b',40),('c',20)]:
            assert backend.set(key,torch.zeros(size,dtype=torch.uint8))
        op.completed_tokens = 3
    monkeypatch.setattr(HiCacheController, '_page_backup', primary)
    c = object.__new__(HybridCacheController)
    c.storage_backend = backend
    c.backup_skip = False
    c.page_size = 1
    op = StorageOperation(torch.tensor([1,2,3]), [1,2,3], hash_value=['a','b','c'],
        pool_transfers=[PoolTransfer(PoolName.MAMBA, host_indices=torch.tensor([1]),keys=['c'])])
    c._page_backup(op)
    assert not backend.exists('c.mamba')
    assert all(backend.exists(key) for key in ['a','b','c'])
    assert op.backup_failure == 'publication_evicted' and op.completed_tokens == 0


def test_two_rank_asymmetric_storage_failures_keep_one_ack_each():
    counts = []
    for fails in (False, True):
        c = object.__new__(HybridCacheController)
        c.storage_stop_event = threading.Event()
        c.backup_queue = Queue()
        c.ack_backup_queue = Queue()
        c.page_size = 1
        op = operation()
        c.register_backup_publication(op)
        c.backup_queue.put(op)
        calls = []
        def write(value):
            calls.append(value.id)
            if fails:
                value.backup_failure = 'sidecar_write'
            else:
                value.completed_tokens = 2
        c._page_backup = write
        put = c.ack_backup_queue.put
        def finish(value):
            put(value)
            c.storage_stop_event.set()
        c.ack_backup_queue.put = finish
        c.backup_thread_func()
        assert len(calls) == (4 if fails else 1)
        counts.append(c.ack_backup_queue.qsize())
        assert c.backup_queue.empty()
    assert counts == [1, 1]


def test_storage_registration_precedes_enqueue_and_failure_releases_pin():
    n = node()
    spec = NS(host_value=torch.tensor([1,2]), hash_value=['a','b'], token_ids=[1,2], prefix_keys=None, comp_xfers={})
    controller = object.__new__(HybridCacheController)
    c = NS(enable_storage=True, tree_core=NS(node_by_id=lambda _:n, build_storage_backup_spec=lambda *a:spec),
           hicache_storage_pass_prefix_keys=False, _build_sidecar_transfers=lambda *a:[],
           inc_host_lock_ref=lambda _:NS(to_dec_params=lambda:'pin'), dec_host_lock_ref=Mock(),
           cache_controller=controller, ongoing_backup={})
    def enqueue(op):
        assert op.id in c.ongoing_backup and op.id in n.mamba_storage_pending
        raise MemoryError()
    controller.backup_queue = NS(put=enqueue)
    controller.ack_backup_queue = Queue()
    n.mamba_loss_logged = True
    UnifiedRadixCache.write_backup_storage(c,n.id)
    c.dec_host_lock_ref.assert_not_called()
    UnifiedRadixCache._finish_storage_backup(c, controller.ack_backup_queue.get_nowait())
    assert not c.ongoing_backup and not n.mamba_storage_pending
    assert n.mamba_storage_failures == {1} and not n.mamba_loss_logged
    c.dec_host_lock_ref.assert_called_once_with(n.id,'pin')


@pytest.mark.parametrize('failure', ['spec', 'enqueue', 'lock'])
def test_asymmetric_preworker_failure_has_matching_terminal_completion(failure):
    peers = []
    for fails in (True, False):
        n = node()
        controller = object.__new__(HybridCacheController)
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        spec = NS(host_value=torch.tensor([1,2]), hash_value=['a','b'], token_ids=[1,2], prefix_keys=None, comp_xfers={})
        c = NS(enable_storage=True, tree_core=NS(node_by_id=lambda _,node=n:node,
            build_storage_backup_spec=lambda *a,spec=spec,fails=fails:None if fails and failure=='spec' else spec),
            hicache_storage_pass_prefix_keys=False,_build_sidecar_transfers=lambda *a:[],
            inc_host_lock_ref=Mock(return_value=NS(to_dec_params=lambda:'pin')),
            dec_host_lock_ref=Mock(),cache_controller=controller,ongoing_backup={})
        if fails and failure=='lock':
            c.inc_host_lock_ref.side_effect=MemoryError()
        if fails and failure=='enqueue':
            controller.backup_queue.put=Mock(side_effect=MemoryError())
        UnifiedRadixCache.write_backup_storage(c,n.id)
        if not fails:
            op=controller.backup_queue.get_nowait()
            op.backup_failure=None
            controller.complete_backup_publication(op)
        peers.append((c,n,controller))
    assert min(controller.ack_backup_queue.qsize() for c,n,controller in peers)==1
    for c,n,controller in peers:
        c.dec_host_lock_ref.assert_not_called()
        UnifiedRadixCache._finish_storage_backup(c,controller.ack_backup_queue.get_nowait())
        assert not c.ongoing_backup and not n.mamba_storage_pending
    assert peers[0][1].mamba_storage_failures
    assert not peers[1][1].mamba_storage_failures


def test_failed_completion_cannot_overtake_older_publication_across_ranks():
    peers = []
    for _ in range(2):
        c = object.__new__(HybridCacheController)
        c.ack_backup_queue = Queue()
        a,b = operation(),operation()
        c.register_backup_publication(a)
        c.register_backup_publication(b)
        peers.append((c,a,b))
    left,a0,b0 = peers[0]
    right,a1,b1 = peers[1]
    b0.backup_failure = 'invalid_spec'
    left.complete_backup_publication(b0)
    right.complete_backup_publication(a1)
    assert min(left.ack_backup_queue.qsize(),right.ack_backup_queue.qsize()) == 0
    left.complete_backup_publication(a0)
    assert min(left.ack_backup_queue.qsize(),right.ack_backup_queue.qsize()) == 1
    assert left.ack_backup_queue.get_nowait() is a0
    assert right.ack_backup_queue.get_nowait() is a1
    assert min(left.ack_backup_queue.qsize(),right.ack_backup_queue.qsize()) == 0
    right.complete_backup_publication(b1)
    assert left.ack_backup_queue.get_nowait() is b0
    assert right.ack_backup_queue.get_nowait() is b1
    assert not left._backup_completions.order and not right._backup_completions.order

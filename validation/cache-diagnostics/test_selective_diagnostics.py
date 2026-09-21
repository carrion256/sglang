import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.mem_cache.cache_diagnostics import CacheDiagnostics
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, PoolName, PoolHitPolicy, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from test_hicache_file_local import storage


def test_cold_misses_quiet_and_counted(caplog):
    d = CacheDiagnostics()
    with caplog.at_level(logging.INFO):
        for _ in range(1000):
            d.record('r', 'cold_miss', requested=220000)
        d.record('r', 'below_threshold', requested=128)
    assert not caplog.records
    assert d.counts['cold_miss'] == 1000


def test_pressure_is_large_only_and_aggregated(caplog):
    clock = [0.0]
    d = CacheDiagnostics(clock=lambda: clock[0])
    with caplog.at_level(logging.INFO):
        d.record('small', 'prefetch_capacity', requested=16000)
        d.record('first', 'prefetch_capacity', requested=220000)
        for _ in range(30):
            d.record('second', 'prefetch_capacity', requested=220000)
        clock[0] = 60
        d.record('third', 'prefetch_capacity', requested=220000)
    assert len(caplog.records) == 2
    assert 'suppressed=30' in caplog.records[-1].message
    assert d.counts['prefetch_capacity'] == 33


@pytest.mark.parametrize('reason', ['lookup_timeout', 'checkpoint_missing', 'peer_miss', 'host_capacity', 'restore_incomplete'])
def test_anomaly_one_event_per_call(reason, caplog):
    d = CacheDiagnostics()
    with caplog.at_level(logging.INFO):
        d.record('r', reason, requested=220000, matched=16000, elapsed=1.25, budget=1)
    assert len(caplog.records) == 1
    assert f'reason={reason}' in caplog.text
    assert 'elapsed_s=1.250' in caplog.text


def test_other_ranks_count_without_duplicate_logs(caplog):
    d = CacheDiagnostics(rank=1)
    with caplog.at_level(logging.INFO):
        d.record('r', 'checkpoint_missing', requested=220000)
    assert not caplog.records
    assert d.counts['checkpoint_missing'] == 1


def test_metrics_have_bounded_labels(monkeypatch):
    import prometheus_client
    registry = prometheus_client.CollectorRegistry()
    original = prometheus_client.Counter
    monkeypatch.setattr(prometheus_client, 'Counter', lambda *a, **kw: original(*a, registry=registry, **kw))
    d = CacheDiagnostics(metrics=True)
    d.record('secret-request-id', 'cold_miss', requested=220000)
    text = prometheus_client.generate_latest(registry).decode()
    assert 'hicache_restore_decisions_total' in text
    assert 'reason="cold_miss"' in text
    assert 'secret-request-id' not in text


def test_inventory_distinguishes_unknown_from_missing_checkpoint():
    keys = ['a', 'b', 'c', 'd']
    inventory = set()
    backend = NS(_get_component_key=lambda k, n=None: f'{k}:{n}',
                 _collect_existing_component_keys=lambda k,t: inventory)
    transfers = [PoolTransfer(PoolName.MAMBA, keys=['d'], hit_policy=PoolHitPolicy.TRAILING_PAGES)]
    cold = HiCacheFile.batch_exists_v2(backend, keys, transfers)
    assert (cold.kv_hit_pages, cold.kv_prefix_pages) == (0, 0)
    inventory.update(f'{k}:None.bin' for k in keys)
    orphan = HiCacheFile.batch_exists_v2(backend, keys, transfers)
    assert (orphan.kv_hit_pages, orphan.kv_prefix_pages) == (0, 4)
    inventory.add(f'b:{PoolName.MAMBA}.bin')
    partial = HiCacheFile.batch_exists_v2(backend, keys, transfers)
    assert (partial.kv_hit_pages, partial.kv_prefix_pages) == (2, 4)


def test_missing_file_quiet_but_bad_read_warns(tmp_path, caplog):
    backend = storage(tmp_path)
    with caplog.at_level(logging.INFO):
        assert backend.get('absent', torch.empty(1, dtype=torch.int32)) is None
    assert not caplog.records
    assert backend.set('short', torch.tensor([1], dtype=torch.int32))
    with caplog.at_level(logging.WARNING):
        assert backend.get('short', torch.empty(2, dtype=torch.int32)) is None
    assert any('Short read' in r.message for r in caplog.records)


@pytest.mark.parametrize('cancelled', [False, True])
@pytest.mark.parametrize('policy', ['timeout', 'best_effort'])
def test_unallocated_lookup_timeout_vs_cancel(cancelled, policy, caplog):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    import time
    op = NS(host_indices=None, storage_hit_count=0, start_time=time.monotonic()-1.5,
            hash_value=[], is_terminated=lambda: cancelled)
    cc = NS(terminate_prefetch=Mock())
    d = CacheDiagnostics()
    cache = NS(ongoing_prefetch={'r': (None, [1]*220000, None, op, None, {})},
               can_terminate_prefetch=lambda op: True, cache_controller=cc,
               _revoke_pending_prefetch=Mock(), cache_diagnostics=d,
               prefetch_timeout_base=1.0, prefetch_timeout_per_page=0.015625,
               prefetch_stop_policy=policy, _prefetch_timeout_check_linear_func=lambda op: True)
    with caplog.at_level(logging.INFO):
        assert UnifiedRadixCache.check_prefetch_progress(cache, 'r') is True
    cc.terminate_prefetch.assert_called_once_with(op)
    cache._revoke_pending_prefetch.assert_called_once_with('r')
    assert len(caplog.records) == (1 if not cancelled and policy == 'timeout' else 0)
    assert d.counts['lookup_timeout'] == (1 if not cancelled and policy == 'timeout' else 0)


def test_no_ongoing_prefetch_is_quiet(caplog):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    with caplog.at_level(logging.INFO):
        assert UnifiedRadixCache.check_prefetch_progress(NS(ongoing_prefetch={}), 'r') is True
    assert not caplog.records


@pytest.mark.parametrize('raw_kv,local_hit,shared_hit,expected', [
    (0, 0, 0, 'cold_miss'),
    (16384, 0, 0, 'checkpoint_missing'),
    (16384, 16384, 0, 'peer_miss'),
    (16384, 16384, 16384, 'host_capacity'),
])
def test_scheduler_query_result_reason(raw_kv, local_hit, shared_hit, expected, caplog):
    from queue import Queue
    import time
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    op = NS(request_id='r', storage_hit_count=shared_hit,
            diagnostic_kv_tokens=raw_kv, diagnostic_local_hit_tokens=local_hit,
            start_time=time.monotonic(), is_terminated=lambda: False)
    q = Queue(); q.put(op)
    cc = NS(prefetch_hit_queue=q, ack_backup_queue=Queue(), host_mem_release_queue=Queue(),
            mem_pool_host=NS(alloc=lambda n: None, available_size=lambda: 0))
    cache = NS(cache_controller=cc, ongoing_prefetch={'r': NS(prefetch_key=[1]*22000)},
               prefetch_threshold=256, page_size=64, cache_diagnostics=CacheDiagnostics(),
               evict_host=Mock(), _revoke_pending_prefetch=Mock())
    with caplog.at_level(logging.INFO):
        UnifiedRadixCache._drain_storage_control_queues_impl(
            cache, n_storage_hit=1, n_backup=0, n_release=0,
            extra_release_counts=None, log_metrics=False)
    cache._revoke_pending_prefetch.assert_called_once_with('r')
    assert cache.cache_diagnostics.counts == {expected: 1}
    assert len(caplog.records) == (0 if expected == 'cold_miss' else 1)


def test_small_restore_large_prefill_correlated_once(caplog):
    d = CacheDiagnostics()
    with caplog.at_level(logging.INFO):
        d.record('r', 'restore_incomplete', requested=2624, matched=2624,
                 details={'pools': [{'pool': 'mamba', 'expected_pages': 1, 'synced_completed_pages': 0}]})
        d.admission('r', 314240, 192512, 1000, 0)
        d.admission('r', 314240, 192512)
    impact = [r.message for r in caplog.records if 'prefill impact' in r.message]
    assert len(impact) == 1
    assert 'admitted_new_tokens=121728' in impact[0]
    assert 'synced_completed_pages' in impact[0]
    assert not d.pending


def test_small_work_and_cold_misses_have_no_impact_log(caplog):
    d = CacheDiagnostics()
    with caplog.at_level(logging.INFO):
        d.record('small', 'checkpoint_missing', requested=2880, matched=512)
        d.admission('small', 14592, 11648)
        d.record('cold', 'cold_miss', requested=400000)
        d.admission('cold', 400000, 0)
    assert not any('prefill impact' in r.message for r in caplog.records)
    assert not d.pending


def test_suppressed_pressure_retains_request_correlation(caplog):
    d = CacheDiagnostics(clock=lambda: 0)
    with caplog.at_level(logging.INFO):
        d.record('first', 'component_capacity', requested=200000)
        d.record('second', 'component_capacity', requested=220000)
        d.admission('second', 220064, 0)
    assert any("req='second'" in r.message and 'prefill impact' in r.message for r in caplog.records)


def test_diagnostic_metadata_is_bounded():
    d = CacheDiagnostics()
    for i in range(5000):
        d.record(str(i), 'backup_pending', requested=1)
    assert len(d.pending) == 4096
    assert '0' not in d.pending
    assert '4999' in d.pending


def test_rank_one_retains_no_request_metadata():
    d = CacheDiagnostics(rank=1)
    d.record('r', 'restore_incomplete', requested=200000)
    assert not d.pending


def test_companion_failure_records_exact_pool_counts(caplog):
    import time
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    transfer = PoolTransfer(PoolName.MAMBA, keys=['private-hash'],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES)
    op = NS(pool_transfers=[transfer],
            pool_storage_result=NS(extra_pool_hit_pages={PoolName.MAMBA: 0}),
            hash_value=['private-hash'], start_time=time.monotonic(), storage_hit_count=64)
    cc = NS(append_host_mem_release=Mock(), prefetch_tokens_occupied=64000)
    cache = NS(_all_reduce_attn_groups=lambda packed, reduce: None,
               page_size=64, cache_controller=cc, dec_host_lock_ref=Mock(),
               ongoing_prefetch={'r': op}, prefetch_loaded_tokens_by_reqid={},
               cache_diagnostics=CacheDiagnostics(), prefetch_timeout_base=1,
               prefetch_timeout_per_page=.015625)
    with caplog.at_level(logging.INFO):
        result = UnifiedRadixCache._sync_and_check_hybrid_prefetch_result(
            cache, 'r', op, 64, op.hash_value, torch.arange(64), 0, None, [0]*64000)
    assert result is None
    details = cache.cache_diagnostics.pending['r']['details']
    assert details['pools'][0]['expected_pages'] == 1
    assert details['pools'][0]['synced_completed_pages'] == 0
    assert details['local_kv_completed'] == 64
    assert 'private-hash' not in caplog.text
    assert cache.prefetch_loaded_tokens_by_reqid['r'] == 0
    cc.append_host_mem_release.assert_called_once()


@pytest.mark.parametrize('accepted', [False, True])
def test_scheduler_admission_uses_full_input_after_host_restore(accepted, caplog):
    import ast
    from pathlib import Path
    source = Path('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py').read_text()
    tree = ast.parse(source)
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'diagnostic_input_tokens' for t in n.targets))
    hook = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.BoolOp)
                and 'diagnostics.admission(' in ast.unparse(n))
    # Select the narrow hook, not its enclosing scheduling branch.
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and 'diagnostics.admission(' in ast.unparse(n)]
    hook = min(candidates, key=lambda n: len(ast.unparse(n)))
    # The inner diagnostics-is-not-None branch depends on its parent setup.
    hook = next(n for n in candidates if 'self.enable_hicache_storage and' in ast.unparse(n.test)
                and 'adder.can_run_list' in ast.unparse(n.test))
    d = CacheDiagnostics(); d.record('r', 'restore_incomplete', requested=64)
    req = NS(rid='r', full_untruncated_fill_ids=[0]*200000,
             prefix_indices=[0]*100000, host_hit_length=10000, storage_hit_length=0)
    env = dict(req=req, self=NS(enable_hicache_storage=True, tree_cache=NS(cache_diagnostics=d)),
               adder=NS(can_run_list=[req] if accepted else []))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<capture>', 'exec'), env)
    req.prefix_indices += [0]*10000
    with caplog.at_level(logging.INFO):
        exec(compile(ast.Module(body=[hook], type_ignores=[]), '<admission>', 'exec'), env)
    if accepted:
        assert 'admitted_new_tokens=90000' in caplog.text
        assert not d.pending
    else:
        assert not caplog.records
        assert 'r' in d.pending

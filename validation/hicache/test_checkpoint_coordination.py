"""Real two-rank CPU collectives with asymmetric cache failures and ownership checks."""
import datetime
import multiprocessing
import traceback
from types import SimpleNamespace as NS
from unittest.mock import Mock
from queue import Queue

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.unified_cache.components import ComponentType, CacheTransferPhase
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV

import pytest
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.checkpoint_coordination import reserve, transfer
from sglang.srt.mem_cache.unified_cache.components.mamba_component import MambaComponent
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


class Slots:
    def __init__(self, failures=0):
        self.failures = failures
        self.next = 10
        self.live = set()
        self.calls = 0

    def alloc(self, count):
        self.calls += 1
        if self.calls <= self.failures:
            return None
        value = torch.arange(self.next, self.next + count, dtype=torch.int64)
        self.next += count
        self.live.update(value.tolist())
        return value

    def available_size(self):
        return 100 - len(self.live)

    def free(self, indices):
        for index in indices.tolist():
            assert index in self.live, (index, self.live)
            self.live.remove(index)


def worker(rank, rendezvous, results):
    try:
        dist.init_process_group('gloo', init_method='file://' + rendezvous,
                                rank=rank, world_size=2, timeout=datetime.timedelta(seconds=15))
        cache = NS(tp_world_size=2, _all_reduce_attn_groups=lambda tensor, op: dist.all_reduce(tensor, op))
        for failures in (0, 1, 2):
            a, b = Slots(), Slots(failures if rank == 0 else 0)
            recovered = []
            def recover():
                assert not a.live and not b.live
                recovered.append(True)
            slots = reserve(cache, ('test', failures), [('a', 1, a.alloc, a.free), ('b', 1, b.alloc, b.free)], recover)
            assert len(recovered) == int(failures > 0)
            assert (slots is None) == (failures == 2)
            if slots:
                a.free(slots[0]); b.free(slots[1])
            assert not a.live and not b.live

        for finished, extra, compressed in ((False,True,False),(False,False,False),
                (False,True,True),(False,False,True),(True,True,False),(True,True,True),
                (True,False,False),(True,False,True)):
            for failure in ((0, 1, 2, 3, 4) if compressed and extra and not finished else (0, 1, 2)):
                active_failures = failure - 2 if failure > 2 else (failure if not compressed else 0)
                active = Slots(active_failures if rank == 0 else 0)
                ckpt = Slots(failure if rank == 0 and compressed and failure <= 2 else 0)
                donated = []
                cursor = torch.tensor([0, 3, 0])
                def donate(req, replacement):
                    donated.append(replacement)
                    return torch.tensor([1])
                pool = NS(mamba_allocator=active, mamba_pool=NS(replayssm_write_pos=cursor, copy_from=Mock()),
                          translate_mamba_indices=lambda x:x, donate_mamba_ping_pong_slot=donate,
                          get_mamba_ping_pong_keep_idx=lambda req:0)
                ckpt.store_from_active = Mock()
                cache.req_to_token_pool = pool; cache.enable_mamba_extra_buffer = extra
                recovery = []
                def evict(params):
                    assert not active.live and not ckpt.live
                    recovery.append(True)
                cache.evict = evict
                comp = NS(cache=cache, int8_ckpt_pool=ckpt if compressed else None)
                req = NS(rid='synthetic', mamba_pool_idx=torch.tensor(1), mamba_last_track_seqlen=64,
                         mamba_ping_pong_track_buffer=torch.tensor([1,2]))
                # The donated old slot is owned by the request, not this reservation.
                old_free = active.free
                active.free = lambda x: None if x.tolist() == [1] else old_free(x)
                params = NS(mamba_value=None)
                length = MambaComponent.prepare_for_caching_req(comp, req, params, 64, finished)
                can_fail = compressed or not finished
                failed = failure in (2, 4) and can_fail
                assert (length == 0) == failed
                assert len(donated) == int(not finished and extra and not failed)
                assert int(cursor[1]) == (0 if finished and not extra and not failed else 3)
                if failed:
                    assert params.mamba_value is None and not active.live and not ckpt.live
                outcomes = [None, None]
                dist.all_gather_object(outcomes, (length, len(donated), len(recovery)))
                assert outcomes[0] == outcomes[1], outcomes

        # Different exact frontiers must skip rather than use MIN to relabel a state.
        req.mamba_last_track_seqlen = 64 + rank
        cache.enable_mamba_extra_buffer = True
        params = NS(mamba_value=None)
        assert MambaComponent.prepare_for_caching_req(comp, req, params, 64, False) == 0
        assert params.mamba_value is None

        for to_host in (True, False):
            for failure in (0, 1, 2):
                full, aux = Slots(), Slots(failure if rank == 0 else 0)
                recovery = []
                cache.evict_host = lambda size: recovery.append(('full', size))
                cache.evict = lambda params: recovery.append(('full', params.num_tokens))
                entry = NS(host_pool=aux, device_pool=aux, device_alloc_fn=None, device_free_fn=None,
                           host_evict_fn=lambda n:recovery.append(('aux',n)),
                           device_evict_fn=lambda n:recovery.append(('aux',n)))
                full.entry_map = {PoolName.MAMBA:entry}
                controller = NS(mem_pool_host=full, mem_pool_device_allocator=full, write_queue=[], load_queue=[], start_writing=Mock())
                pool = PoolTransfer(name=PoolName.MAMBA,
                                    device_indices=torch.tensor([3]) if to_host else None,
                                    host_indices=None if to_host else torch.tensor([3]))
                result = transfer(controller, cache, torch.tensor([1,2]), [pool], 1, None, to_host=to_host)
                assert (result is None) == (failure == 2)
                assert len(controller.write_queue if to_host else controller.load_queue) == int(failure < 2)
                assert controller.start_writing.call_count == int(to_host and failure < 2)
                if result is None:
                    assert not full.live and not aux.live
                outcomes = [None,None]
                dist.all_gather_object(outcomes, (result is None, recovery))
                assert outcomes[0] == outcomes[1]
        # Drive actual scheduler cache methods, not only the reservation helper.
        for fault in ('none', 'spec', 'sidecar', 'allocation', 'enqueue'):
            host = Slots(2 if rank == 0 and fault == 'allocation' else 0)
            c = object.__new__(UnifiedRadixCache)
            c.tp_world_size = 2; c.tp_group = dist.group.WORLD
            c.attn_cp_group = None; c.attn_tp_group = None
            c.tree_core = NS(enable_storage=True)
            c.prefetch_threshold = 1; c.page_size = 1
            c.ongoing_prefetch = {}
            c.inc_host_lock_ref = lambda _: NS(to_dec_params=lambda:'pin')
            c.dec_host_lock_ref = Mock()
            c.evict_host = Mock()
            c.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
            component = NS(cache=c, _mamba_pool_host=host)
            component.prepare_prefetch = lambda node_id, **kw: MambaComponent.prepare_prefetch(component,node_id,**kw)
            c.components = {ComponentType.MAMBA:component}
            def spec(*args, **kw):
                if rank == 0 and fault == 'spec':
                    raise MemoryError()
                return [PoolTransfer(name=PoolName.MAMBA, host_indices=kw['host_indices'])]
            c.tree_core = NS(page_size=1, enable_storage=True, prefetch_anchor_info=lambda _: (None,None), is_eagle=False, build_hicache_transfers=spec)
            def sidecars(*args):
                if rank == 0 and fault == 'sidecar':
                    raise MemoryError()
                return []
            c._build_sidecar_transfers = sidecars
            queue = Queue()
            if rank == 0 and fault == 'enqueue':
                queue._put = Mock(side_effect=MemoryError())
            c.cache_controller = NS(prefetch_rate_limited=lambda:False, prefetch_queue=queue, prefetch_tokens_occupied=0)
            c.prefetch_from_storage('synthetic', 1, [1,2,3])
            assert len(c.ongoing_prefetch) == int(fault == 'none')
            assert queue.qsize() == int(fault == 'none')
            assert c.dec_host_lock_ref.call_count == int(fault != 'none')
            assert len(host.live) == int(fault == 'none')

            if fault == 'none':
                op = queue.get_nowait()
                op.storage_hit_count = 3; op.hash_value = ['a','b','c']
                full = Slots()
                cc = c.cache_controller
                cc.mem_pool_host = full
                cc.prefetch_hit_queue = Queue(); cc.prefetch_hit_queue.put(op)
                cc.prefetch_buffer = Queue()
                if rank == 0:
                    cc.prefetch_buffer._put = Mock(side_effect=MemoryError())
                cc.ack_backup_queue = Queue(); cc.host_mem_release_queue = Queue()
                cc.extra_host_mem_release_queues = {}
                cc.append_host_mem_release = lambda **kw:[host.free(x.host_indices) for x in kw.get('extra_pools', [])]
                c._drain_storage_control_queues_impl(1,0,0,{},False)
                assert not full.live and not host.live and not c.ongoing_prefetch
                assert cc.prefetch_buffer.qsize() == 0

        for fault in ('none', 'spec', 'sidecar'):
            c = object.__new__(UnifiedRadixCache)
            c.tp_world_size = 2; c.tp_group = dist.group.WORLD
            c.attn_cp_group = None; c.attn_tp_group = None
            c.inc_host_lock_ref = lambda _: NS(to_dec_params=lambda:'pin')
            c.dec_host_lock_ref = Mock()
            def spec(_):
                if rank == 0 and fault == 'spec':
                    raise MemoryError()
                return torch.tensor([1,2]), {}
            def sidecars(*args):
                if rank == 0 and fault == 'sidecar':
                    raise MemoryError()
                return []
            c.tree_core = NS(build_backup_spec=spec, commit_backup=Mock())
            c._build_backup_sidecar = sidecars
            c._execute_kv_backup = Mock(return_value=torch.tensor([3,4]))
            c._track_write_through_node = Mock()
            success = c._execute_and_commit_kv_backup(BackupKV([1]), write_back=True)
            assert success == (fault == 'none')
            assert c.tree_core.commit_backup.call_count == int(success)
            assert c.dec_host_lock_ref.call_count == int(not success)

        # Restore preparation failures release both ancestor pins and request slots.
        for fault in ('none', 'spec', 'sidecar', 'allocation'):
            c = object.__new__(UnifiedRadixCache)
            c.tp_world_size = 2; c.tp_group = dist.group.WORLD
            c.attn_cp_group = None; c.attn_tp_group = None
            c.inc_host_lock_ref = lambda _:NS(to_dec_params=lambda:'host')
            c.inc_lock_ref = lambda _:NS(delta=0,to_dec_params=lambda:'device')
            c.dec_host_lock_ref = Mock(); c.dec_lock_ref = Mock(); c.evict = Mock()
            c.load_back_threshold = 0
            active = Slots(2 if rank == 0 and fault == 'allocation' else 0)
            c.req_to_token_pool = NS(mamba_allocator=active)
            req = NS(mamba_pool_idx=None)
            def spec(*args, **kw):
                if rank == 0 and fault == 'spec':
                    raise MemoryError()
                return PoolTransfer(name=PoolName.KV,host_indices=torch.tensor([1,2])), {}
            def sidecars(*args):
                if rank == 0 and fault == 'sidecar':
                    raise MemoryError()
                return []
            c.tree_core = NS(component_has_host_value_only=lambda *a:True,
                             build_load_back_spec=spec,commit_load_back=Mock(return_value=[]))
            c._build_sidecar_transfers = sidecars
            comp = NS(cache=c, tree_core=c.tree_core, component_type=ComponentType.MAMBA)
            comp.prepare_load_back = lambda *a, **kw:MambaComponent.prepare_load_back(comp,*a,**kw)
            comp.finalize_load_back = lambda *a, **kw:MambaComponent.finalize_load_back(comp,*a,**kw)
            c._components_tuple = (comp,)
            c._apply_cache_actions = Mock(); c.ongoing_load_back = {}
            c.cache_controller = NS(load=Mock(return_value=torch.tensor([5,6])))
            success = c.load_back(1, req=req)
            assert success == (fault == 'none')
            if not success:
                assert req.mamba_pool_idx is None and not active.live
                c.dec_host_lock_ref.assert_called_once_with(1,'host')
            c.dec_lock_ref.assert_called_once_with(1,'device')

        import io
        import logging
        from test_checkpoint_backup import test_real_cpu_tree_eviction_preserves_checkpoint_or_continues
        capture = io.StringIO()
        handler = logging.StreamHandler(capture)
        logging.getLogger().addHandler(handler)
        class Captured:
            @property
            def text(self):
                return capture.getvalue()
        def configure(c, controller, target, events, case):
            c.tp_world_size = 2; c.tp_group = dist.group.WORLD
            c.attn_cp_group = None; c.attn_tp_group = None
            full, aux = Slots(), Slots(100 if rank == 0 and case.endswith('failure') else 0)
            full.entry_map = {PoolName.MAMBA:NS(host_pool=aux,
                host_evict_fn=lambda count:c.evict_host(count,ComponentType.MAMBA))}
            controller.mem_pool_host = full; controller.write_queue = []
            def start():
                controller.write_queue.clear()
                controller.ack_write_queue.append(NS(finish_event=NS(synchronize=lambda:events.append('copy_complete')),node_ids=[target.id]))
            controller.start_writing = start
            controller.write = lambda values, node_id, extra_pools:transfer(controller,c,values,extra_pools,node_id,None,to_host=True)
        try:
            for case in ('internal','internal_failure','partial_leaf','partial_leaf_failure','path_cap','split_pending'):
                with pytest.MonkeyPatch.context() as patch:
                    test_real_cpu_tree_eviction_preserves_checkpoint_or_continues(patch,Captured(),case,configure)
        finally:
            logging.getLogger().removeHandler(handler)

        results.put((rank, None))
    except BaseException:
        results.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_real_two_rank_failure_recovery_and_commit(tmp_path):
    ctx = multiprocessing.get_context('fork')
    results = ctx.Queue()
    children = [ctx.Process(target=worker, args=(rank, str(tmp_path/'gloo'), results)) for rank in range(2)]
    for child in children:
        child.start()
    try:
        reports = [results.get(timeout=40) for _ in children]
        assert all(error is None for rank, error in reports), "\n".join(error for rank, error in reports if error)
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)

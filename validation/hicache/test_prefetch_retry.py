"""CPU control tests; synthetic cache results, no inference or production data."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.mem_cache.prefetch_retry import retry_advanced_prefix


def fixture(previous=0, current=128, length=1025):
    cache = NS(is_write_back=True, tree_core=object(), prefetch_threshold=256,
               tp_world_size=1)
    probes, submissions = [], []
    req = NS(rid='synthetic', _hicache_prefetch_prefix=previous, output_ids=[],
             storage_hit_length=0, full_untruncated_fill_ids=list(range(length)),
             prefix_indices=[], host_hit_length=0, last_host_node=3,
             extra_key=None, cache_salt=None)
    def probe(tree, cow_mamba=None):
        probes.append(cow_mamba)
        req.prefix_indices = list(range(current))
    req.init_next_round_input = probe
    req._compute_max_prefix_len = lambda n: n-1
    # Execute the real patched enqueue helper, including captured prefix and slice.
    source = Path('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prefetch_kvcache')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    env = {}
    exec(compile(ast.fix_missing_locations(module), '<enqueue>', 'exec'), env)
    cache.is_backuped = lambda node: True
    cache.is_root = lambda node: False
    cache.hicache_storage_pass_prefix_keys = False
    cache.get_last_hash_value = lambda node: 'synthetic-anchor'
    cache.prefetch_from_storage = lambda *a, **kw: submissions.append((a,kw))
    scheduler = NS(tree_cache=cache, enable_hicache_storage=True)
    scheduler._prefetch_kvcache = lambda r: env['_prefetch_kvcache'](scheduler,r)
    return scheduler,req,probes,submissions


def test_retry_uses_new_anchor_and_suffix_once():
    scheduler,req,probes,submissions = fixture()
    assert retry_advanced_prefix(scheduler,req)
    assert probes == [False, False]
    args,kw = submissions[0]
    assert args[1] == 3 and args[2] == list(range(128,1024))
    assert args[3] == 'synthetic-anchor'
    assert req._hicache_prefetch_prefix == 128
    assert req._hicache_prefetch_retried
    assert not retry_advanced_prefix(scheduler,req)
    assert len(submissions)==1


@pytest.mark.parametrize('previous,current,length', [(128,128,1025),(256,128,1025),(0,128,300)])
def test_no_advance_or_short_tail(previous,current,length):
    scheduler,req,_,submissions=fixture(previous,current,length)
    assert not retry_advanced_prefix(scheduler,req)
    assert not submissions and not getattr(req,'_hicache_prefetch_retried',False)


@pytest.mark.parametrize('field,value', [('_hicache_prefetch_prefix',None),
    ('_hicache_prefetch_retried',True),('_hicache_retry_admitted',True),
    ('output_ids',[1]),('storage_hit_length',64)])
def test_lifetime_guards(field,value):
    scheduler,req,probes,submissions=fixture()
    setattr(req,field,value)
    assert not retry_advanced_prefix(scheduler,req)
    assert not probes and not submissions


def test_non_unified_cache():
    scheduler,req,probes,_=fixture()
    del scheduler.tree_cache.tree_core
    assert not retry_advanced_prefix(scheduler,req)
    assert not probes


def test_admission_hook_order():
    source=Path('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py').read_text()
    assert source.index('prefetch_done = self.tree_cache.check_prefetch_progress') < source.index('if retry_advanced_prefix(self, req):')
    assert source.index('if retry_advanced_prefix(self, req):') < source.index('diagnostic_input_tokens = len(req.full_untruncated_fill_ids)')
    assert 'req._hicache_retry_admitted = True' in source


def worker(rank,path):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        # Both advance; only one advances; both advance to different boundaries;
        # one has consumed retry; then successful agreement again.
        for case in range(5):
            current = 128
            if case==1 and rank==1: current=0
            if case==2 and rank==1: current=256
            scheduler,req,_,submissions=fixture(current=current)
            cache=scheduler.tree_cache
            cache.tp_world_size=2
            cache._all_reduce_attn_groups=lambda tensor,op: dist.all_reduce(tensor,op=op)
            if case==3 and rank==1: req._hicache_prefetch_retried=True
            result=retry_advanced_prefix(scheduler,req)
            assert result == (case in (0,4))
            assert len(submissions)==int(result)
    finally:
        dist.destroy_process_group()


def test_two_rank_retry_agreement(tmp_path):
    torch.multiprocessing.spawn(worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)


def test_declined_retry_does_not_loop_or_retain_old_diagnostic():
    scheduler,req,_,submissions=fixture()
    scheduler.tree_cache.cache_diagnostics=NS(pending={req.rid: {'reason':'cold_miss'}})
    scheduler._prefetch_kvcache=lambda r: None
    assert retry_advanced_prefix(scheduler,req)
    assert req.rid not in scheduler.tree_cache.cache_diagnostics.pending
    assert not retry_advanced_prefix(scheduler,req)
    assert not submissions

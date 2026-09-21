"""CPU regression coverage for prefetch namespace preservation and rank agreement."""
import ast
import datetime
import importlib.util
import logging
import multiprocessing
from pathlib import Path
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey

OMITTED = object()


def cache_fixture(backend, anchor_ns=(None, None), root=True, bigram=False):
    anchor = NS(key=RadixKey([], extra_key=anchor_ns[0], cache_salt=anchor_ns[1]), protect_host=Mock())
    if backend == 'unified':
        c = object.__new__(UnifiedRadixCache)
        c.tree_core = NS(enable_storage=True, page_size=4, is_eagle=bigram,
                         prefetch_anchor_info=lambda _: anchor_ns, is_root=lambda _:root)
        c.tree_components = ()
        c._build_sidecar_transfers = Mock(return_value=[])
        c.inc_host_lock_ref = Mock(return_value=NS(to_dec_params=lambda:'pin'))
        c.dec_host_lock_ref = Mock()
        cc = NS(prefetch_rate_limited=lambda:False, prefetch_queue=Queue(), prefetch_tokens_occupied=0)
        node = 1
    else:
        c = object.__new__(HiRadixCache)
        c.enable_storage = True
        c.root_node = anchor if root else object()
        c.is_eagle = bigram
        c.page_size = 4
        c._get_extra_pools = lambda:{}
        cc = NS(prefetch_rate_limited=lambda:False, prefetch=Mock(return_value=object()), prefetch_tokens_occupied=0)
        node = anchor
    c.cache_controller = cc
    c.tp_world_size = 1
    c.attn_cp_group = c.attn_tp_group = None
    c.prefetch_threshold = 1
    c.ongoing_prefetch = {}
    return c, node, anchor


def invoke(c, node, namespace):
    kwargs = {} if namespace is OMITTED else {'request_namespace':namespace}
    c.prefetch_from_storage('synthetic', node, list(range(11)), None, None, **kwargs)
    pending = c.ongoing_prefetch.get('synthetic')
    return None if pending is None else pending[1]


@pytest.mark.parametrize('backend', ['unified','hiradix'])
@pytest.mark.parametrize('namespace', [(None,'salt-a'),('adapter-a',None),('adapter-a','salt-a'),(None,None),(None,'')])
@pytest.mark.parametrize('bigram', [False,True])
def test_cold_root_preserves_requested_namespace(backend, namespace, bigram):
    c,node,_ = cache_fixture(backend,bigram=bigram)
    key = invoke(c,node,namespace)
    assert (key.extra_key,key.cache_salt)==(namespace[0],namespace[1] or None)
    expected=RadixKey(range(11),extra_key=namespace[0],cache_salt=namespace[1],is_bigram=bigram).page_aligned(4)
    assert key.child_key(4)==expected.child_key(4)
    assert list(key)==list(expected)
    assert c.cache_controller.prefetch_tokens_occupied==len(expected)


@pytest.mark.parametrize('backend', ['unified','hiradix'])
@pytest.mark.parametrize('namespace', [OMITTED,('adapter','salt')])
def test_nonroot_and_legacy_anchor_namespace(backend,namespace):
    c,node,_=cache_fixture(backend,('adapter','salt'),root=False)
    key=invoke(c,node,namespace)
    assert (key.extra_key,key.cache_salt)==('adapter','salt')


@pytest.mark.parametrize('backend', ['unified','hiradix'])
@pytest.mark.parametrize('namespace', [(None,None),('adapter','other'),('other','salt')])
def test_conflicting_anchor_has_no_ownership_side_effects(backend,namespace,caplog):
    c,node,anchor=cache_fixture(backend,('adapter','salt'),root=False)
    assert invoke(c,node,namespace) is None
    assert c.cache_controller.prefetch_tokens_occupied==0
    assert c.ongoing_prefetch=={}
    assert 'reason=namespace_mismatch' in caplog.text
    if backend=='unified':
        c.inc_host_lock_ref.assert_not_called()
        c._build_sidecar_transfers.assert_not_called()
        assert c.cache_controller.prefetch_queue.empty()
    else:
        anchor.protect_host.assert_not_called()
        c.cache_controller.prefetch.assert_not_called()


def test_actual_host_tree_rematches_only_requested_namespace(monkeypatch):
    path='/sgl-workspace/sglang/test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py'
    spec=importlib.util.spec_from_file_location('namespace_tree_fixture',path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'get_device',lambda:'cpu')
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    cfg=module.CacheConfig(components=(ComponentType.FULL,),num_layers=1,full_attention_layer_ids=(0,),kv_size=32,max_context_len=32,max_num_reqs=2)
    cache,_,_=module.build_fixture(cfg)
    tree=cache.tree_core
    tree.set_hicache_enabled()
    for salt in ('salt-a','salt-b'):
        c,node,_=cache_fixture('unified')
        key=invoke(c,node,('adapter',salt))
        tree.insert_host(tree.root_node.id,key,torch.arange(len(key)),['hash']*len(key))
    assert len(tree.root_node.children)==2
    for salt in ('salt-a','salt-b','other',None):
        key=RadixKey(list(range(8)),extra_key='adapter',cache_salt=salt)
        result=cache.match_prefix(MatchPrefixParams(key=key))
        assert result.host_hit_length==(8 if salt in ('salt-a','salt-b') else 0)
    assert cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8)),extra_key='other',cache_salt='salt-a'))).host_hit_length==0


def load_method(relative, name):
    path=Path('/sgl-workspace/sglang/python/sglang/srt')/relative
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    scope={'logger':logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope[name]


@pytest.mark.parametrize('caller',['scheduler','decode'])
def test_actual_callers_forward_namespace(caller):
    node=NS(backuped=True,get_last_hash_value=lambda:None)
    tree=NS(is_backuped=lambda _:True,is_root=lambda _:True,
            get_last_hash_value=lambda _:None,hicache_storage_pass_prefix_keys=False,
            prefetch_from_storage=Mock(),resolve_node_handle=lambda _:node,ongoing_prefetch={})
    req=NS(rid='synthetic',extra_key='adapter',cache_salt='salt',last_host_node=1,
           prefix_indices=[],host_hit_length=0,full_untruncated_fill_ids=list(range(12)),
           origin_input_ids=list(range(12)),init_next_round_input=Mock(),_compute_max_prefix_len=lambda n:n-1)
    if caller=='scheduler':
        method=load_method('managers/scheduler.py','_prefetch_kvcache')
        method(NS(enable_hicache_storage=True,tree_cache=tree),req)
    else:
        method=load_method('disaggregation/decode_hicache_mixin.py','_start_hicache_prefetch')
        method(NS(tree_cache=tree),req,NS(l3_storage_hit_length=8,last_host_node=1,l1_prefix_len=0,l2_host_hit_length=0))
    assert tree.prefetch_from_storage.call_count==1
    assert tree.prefetch_from_storage.call_args.kwargs=={'request_namespace':('adapter','salt')}


def rank_worker(rank,rendezvous,results):
    try:
        dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=15))
        for backend in ('unified','hiradix'):
            for scenario in ('success','asymmetric_anchor','rank_namespace'):
                anchor_ns=('adapter','salt') if rank==0 or scenario!='asymmetric_anchor' else ('adapter','wrong')
                c,node,anchor=cache_fixture(backend,anchor_ns,root=scenario!='asymmetric_anchor')
                c.tp_world_size=2
                c._all_reduce_attn_groups=lambda tensor,op:dist.all_reduce(tensor,op)
                requested=('adapter','different' if rank==1 and scenario=='rank_namespace' else 'salt')
                key=invoke(c,node,requested)
                assert (key is not None)==(scenario=='success')
                if key is None:
                    assert c.cache_controller.prefetch_tokens_occupied==0
                    if backend=='unified':c.inc_host_lock_ref.assert_not_called()
                    else:anchor.protect_host.assert_not_called()
        results.put(None)
    except Exception:
        import traceback
        results.put(traceback.format_exc())
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def test_two_rank_namespace_rejection_is_consistent(tmp_path):
    ctx=multiprocessing.get_context('spawn');results=ctx.Queue()
    processes=[ctx.Process(target=rank_worker,args=(r,str(tmp_path/'rendezvous'),results)) for r in range(2)]
    for p in processes:p.start()
    try:
        for p in processes:p.join(60)
        assert all(not p.is_alive() and p.exitcode==0 for p in processes)
        assert [results.get(timeout=5) for _ in processes]==[None,None]
    finally:
        for p in processes:
            if p.is_alive():p.terminate();p.join(10)


@pytest.mark.parametrize('bigram', [False, True])
def test_namespaces_do_not_change_chained_disk_hashes(bigram):
    keys=[RadixKey(range(13),extra_key=extra,cache_salt=salt,is_bigram=bigram)
          for extra,salt in [(None,None),(None,''),(None,'salt'),('adapter','salt')]]
    chains=[]
    for key in keys:
        prior=None;chain=[]
        for start in range(0,12,4):
            prior=key.hash_page(start,start+4,prior)
            chain.append(prior)
        chains.append(chain)
    assert all(chain==chains[0] for chain in chains)

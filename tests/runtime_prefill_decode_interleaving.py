"""CPU scheduler selection tests; GPU execution/allocators are explicit doubles."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(os.environ.get('INTERLEAVE_CANDIDATE', '/sgl-workspace/sglang/python/sglang/srt'))


class Mode:
    def __init__(self, name): self.name = name
    def is_extend(self): return self.name == 'P'
    def is_decode(self): return self.name == 'D'


class Request:
    def __init__(self):
        self.prefix_indices = []
        self.extend_range = NS(end=0)


class Batch:
    def __init__(self, reqs=None, mode='D', chunked_req=None):
        self.reqs = list(reqs or [])
        self.forward_mode = Mode(mode)
        self.chunked_req = chunked_req
        self.is_prefill_only = False
        self.batch_is_full = False
    def is_empty(self): return not self.reqs
    def batch_size(self): return len(self.reqs)
    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [r for r in self.reqs if r not in chunked_req_to_exclude]
    def merge_batch(self, other): self.reqs += other.reqs


def methods(root):
    tree = ast.parse((root/'managers/scheduler.py').read_text())
    names = {'get_next_batch_to_run', 'get_new_batch_prefill', '_get_new_batch_prefill_raw'}
    nodes = [copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
        if n.name == '_get_new_batch_prefill_raw':
            # Execute real housekeeping and gate, stub only downstream admission.
            end = next(i for i,x in enumerate(n.body) if isinstance(x,ast.If) and ast.unparse(x.test)=='self.enable_priority_preemption or self.is_hybrid_swa')
            n.body = n.body[:end] + ast.parse('return self.admit(running_batch)').body
    ns = dict(NextBatchPlan=lambda **kw: NS(**kw),
              get_memory=lambda: NS(enable_flexkv=False),
              set_schedule_time_batch=lambda batch: None)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual scheduler>', 'exec'), ns)
    return {name:ns[name] for name in names}


class SchedulerHarness:
    def __init__(self, limit, root=ROOT):
        for name, method in methods(root).items():
            setattr(self, name, method.__get__(self))
        self.prefill_batches_before_decode = limit
        self._consecutive_prefill_batches = 0
        self._interleave_decode_pending = False
        self.enable_fpm = False
        self.dllm_config = None
        self.enable_hisparse = False
        self.require_mlp_sync = False
        self.prefill_delayer = None
        self.enable_hierarchical_cache = True
        self.chunked_req = Request()
        self.prefill_available = True
        self.admissions = self.stashes = self.grammar_checks = self.cache_checks = 0
        self.cancel = self.remove_decoders = self.finish_prefill = False
        self.queued_grammar = []
        self.grammar_manager = NS(has_waiting_grammars=self.grammar_ready,
                                  get_ready_grammar_requests=lambda: [])
        self.tree_cache = NS(check_hicache_events=self.cache_events)
        self.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda batch,**kw:batch)
        self.ngram_embedding_manager = NS(prepare_for_forward=lambda batch,**kw:batch)
        self._abort_on_waiting_timeout = lambda:None
        self._abort_on_running_timeout = lambda batch:None
        self.running = Batch([Request()])
        self.last = None
    def grammar_ready(self):
        self.grammar_checks += 1
        return True
    def cache_events(self): self.cache_checks += 1
    def process_pending_chunked_abort(self):
        if self.cancel:
            self.chunked_req = None
            self.prefill_available = False
            self.cancel = False
            # Result/cancellation internals are validated separately, not simulated here.
            self.last = None
    def stash_chunked_request(self, req):
        self.stashes += 1
        req.prefix_indices = list(range(req.extend_range.end))
    def admit(self, running):
        if not self.prefill_available: return None,running
        self.admissions += 1
        req = self.chunked_req or Request()
        req.extend_range.end = len(req.prefix_indices)+64
        if self.finish_prefill:
            self.chunked_req = None
            self.prefill_available = False
            return Batch([req], 'P'),running
        self.chunked_req = req
        return Batch([req], 'P', req),running
    def update_running_batch(self, batch):
        if self.remove_decoders: batch.reqs=[]
        batch.forward_mode = Mode('D')
        return batch
    def step(self):
        if self.cancel:
            self.process_pending_chunked_abort()
        result = self.get_next_batch_to_run(self.running, self.last)
        self.running = result.running_batch
        self.last = result.batch_to_run
        return self.last.forward_mode.name if self.last else '-'


class SchedulingTests(unittest.TestCase):
    def test_exact_ratios(self):
        for n in (1,2,3,7):
            h=SchedulerHarness(n)
            self.assertEqual(''.join(h.step() for _ in range((n+1)*3)), ('P'*n+'D')*3)
            self.assertEqual(h.admissions,n*3)
            self.assertEqual(h.grammar_checks,(n+1)*3)
            self.assertEqual(h.cache_checks,(n+1)*3)
    def test_zero_is_legacy(self):
        for available in (True,False):
            current=SchedulerHarness(0)
            original=SchedulerHarness(0,Path(os.environ['INTERLEAVE_BASELINE']))
            current.prefill_available=original.prefill_available=available
            self.assertEqual([current.step() for _ in range(10)], [original.step() for _ in range(10)])
    def test_no_decoders_never_waits(self):
        h=SchedulerHarness(2);h.running=Batch()
        self.assertEqual(''.join(h.step() for _ in range(10)), 'P'*10)
        self.assertEqual(h._consecutive_prefill_batches,2)
        h.running=Batch([Request()])
        self.assertEqual(h.step(),'D')
    def test_no_prefill_never_waits(self):
        h=SchedulerHarness(2);h.prefill_available=False
        self.assertEqual(''.join(h.step() for _ in range(10)), 'D'*10)
    def test_no_work_does_not_count(self):
        h=SchedulerHarness(2);h.prefill_available=False;h.running=Batch()
        self.assertEqual(h.step(),'-');self.assertEqual(h._consecutive_prefill_batches,0)
    def test_decoder_disappears_retry_prefill_same_step(self):
        h=SchedulerHarness(2)
        self.assertEqual(h.step()+h.step(),'PP')
        h.remove_decoders=True
        self.assertEqual(h.step(),'P')
        self.assertEqual(h.admissions,3)
        self.assertEqual(h._consecutive_prefill_batches,2)
    def test_failed_prefill_admission_does_not_count(self):
        h=SchedulerHarness(2);h.prefill_available=False
        self.assertEqual(h.step(),'D');self.assertEqual(h._consecutive_prefill_batches,0)
        h.prefill_available=True
        self.assertEqual(h.step()+h.step()+h.step(),'PPD')
    def test_chunk_stashed_once_and_not_decoded(self):
        h=SchedulerHarness(1);req=h.chunked_req
        self.assertEqual(h.step(),'P')
        self.assertEqual(h.step(),'D')
        self.assertEqual(h.stashes,1);self.assertNotIn(req,h.running.reqs)
        self.assertEqual(h.step(),'P');self.assertEqual(h.stashes,1)
        self.assertEqual(h.step(),'D');self.assertEqual(h.stashes,2)
    def test_final_chunk_merges_once(self):
        h=SchedulerHarness(2);req=h.chunked_req
        h.step();h.finish_prefill=True;h.step()
        self.assertEqual(h.step(),'D')
        self.assertEqual(h.running.reqs.count(req),1)
        h.step();self.assertEqual(h.running.reqs.count(req),1)
    def test_prefill_only_batch_not_given_decode_turn(self):
        h=SchedulerHarness(1);h.running.is_prefill_only=True
        self.assertEqual(h.step()+h.step()+h.step(),'PPP')
    def test_parked_cancellation_not_resumed(self):
        h=SchedulerHarness(1);h.step();h.step();h.cancel=True
        self.assertEqual(h.step(),'D');self.assertIsNone(h.chunked_req)
        self.assertEqual(h.admissions,1)
    def test_idle_batch_does_not_count(self):
        h=SchedulerHarness(2)
        h.ngram_embedding_manager.prepare_for_forward=lambda batch,**kw:Batch([Request()], 'I')
        self.assertEqual(h.step(),'I')
        self.assertEqual(h._consecutive_prefill_batches,0)
    def test_partial_retraction_keeps_decode_turn(self):
        h=SchedulerHarness(1);h.running.reqs.append(Request());h.step()
        def retract(batch):
            batch.reqs.pop()
            batch.forward_mode=Mode('D')
            return batch
        h.update_running_batch=retract
        self.assertEqual(h.step(),'D')
        self.assertEqual(h.running.batch_size(),1)
        self.assertEqual(h._consecutive_prefill_batches,0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        t=ast.parse((ROOT/'server_args.py').read_text())
        n=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_check_prefill_decode_interleaving')
        ns={};exec(compile(ast.Module(body=[n],type_ignores=[]),'<actual validation>','exec'),ns)
        self.check=ns[n.name]
        self.args=dict(prefill_batches_before_decode=2,tp_size=2,pp_size=1,dp_size=1,
                       enable_dp_attention=False,dcp_size=1,disaggregation_mode='null',
                       enable_mixed_chunk=False,enable_pdmux=False,dllm_algorithm=None,
                       enable_hisparse=False,enable_two_batch_overlap=False,
                       disable_overlap_schedule=False,chunked_prefill_size=6144,is_embedding=False)
    def test_enabled_valid(self): self.check(NS(**self.args))
    def test_negative_rejected(self):
        self.args['prefill_batches_before_decode']=-1
        with self.assertRaises(ValueError):self.check(NS(**self.args))
    def test_unsupported_modes_rejected_only_when_enabled(self):
        changes=dict(tp_size=1,pp_size=2,dp_size=2,enable_dp_attention=True,dcp_size=2,
                     disaggregation_mode='prefill',enable_mixed_chunk=True,enable_pdmux=True,
                     dllm_algorithm='test',enable_hisparse=True,enable_two_batch_overlap=True,
                     disable_overlap_schedule=True,chunked_prefill_size=-1,is_embedding=True)
        for key,value in changes.items():
            with self.subTest(key=key):
                a={**self.args,key:value}
                with self.assertRaises(ValueError):self.check(NS(**a))
                a['prefill_batches_before_decode']=0
                self.check(NS(**a))


if __name__=='__main__':unittest.main()

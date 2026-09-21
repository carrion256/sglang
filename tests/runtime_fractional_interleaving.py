"""Exercise fractional scheduling with the real scheduler-method CPU harness."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import runtime_prefill_decode_interleaving as harness


class FractionalTests(unittest.TestCase):
    def test_exact_reciprocal_sequences(self):
        for denominator in (2,3,4,5,10,100):
            with self.subTest(denominator=denominator):
                h=harness.SchedulerHarness(1/denominator)
                self.assertEqual(''.join(h.step() for _ in range((denominator+1)*3)),('P'+'D'*denominator)*3)
    def test_no_decoders_no_idle_and_no_accumulating_debt(self):
        h=harness.SchedulerHarness(.5);h.running.reqs=[]
        self.assertEqual(''.join(h.step() for _ in range(8)),'P'*8)
        h.running.reqs=[harness.Request()]
        self.assertEqual(''.join(h.step() for _ in range(3)),'DDP')
    def test_no_prefill_continue_decode(self):
        h=harness.SchedulerHarness(.5);h.prefill_available=False
        self.assertEqual(''.join(h.step() for _ in range(8)),'D'*8)
        h.prefill_available=True
        self.assertEqual(''.join(h.step() for _ in range(3)),'PDD')
    def test_idle_does_not_spend_credit(self):
        h=harness.SchedulerHarness(.5);h.step();h.running.reqs=[];h.prefill_available=False
        self.assertEqual(h.step(),'-');self.assertEqual(h._consecutive_prefill_batches,2)
    def test_empty_decode_retries_prefill(self):
        h=harness.SchedulerHarness(.5);h.step();h.remove_decoders=True
        self.assertEqual(h.step(),'P')
    def test_parked_chunk_not_stashed_twice(self):
        h=harness.SchedulerHarness(.1);req=h.chunked_req;h.step()
        self.assertEqual(''.join(h.step() for _ in range(10)),'D'*10)
        self.assertEqual(h.stashes,1);self.assertNotIn(req,h.running.reqs)
    def test_final_chunk_merges_once(self):
        h=harness.SchedulerHarness(.5);req=h.chunked_req;h.finish_prefill=True
        h.step();h.step();h.step()
        self.assertEqual(h.running.reqs.count(req),1)
    def test_cancel_parked_chunk(self):
        h=harness.SchedulerHarness(.5);h.step();h.cancel=True
        self.assertEqual(''.join(h.step() for _ in range(4)),'DDDD')
        self.assertIsNone(h.chunked_req);self.assertEqual(h.admissions,1)
    def test_validation_ratios(self):
        base=harness.ValidationTests();base.setUp()
        for ratio in (.5,.1,1/3,0,1,2,7):
            base.check(NS(**{**base.args,'prefill_batches_before_decode':ratio}))
        for ratio in (-.5,float('nan'),float('inf'),1.5,.3,1e-320):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                base.check(NS(**{**base.args,'prefill_batches_before_decode':ratio}))
    def test_cli_float(self):
        from sglang.srt.server_args import ServerArgs
        parser=argparse.ArgumentParser();ServerArgs.add_cli_args(parser)
        for value in ('0.5','0.1','2'):
            args=parser.parse_args(['--model-path','synthetic','--prefill-batches-before-decode',value])
            self.assertEqual(args.prefill_batches_before_decode,float(value))


if __name__=='__main__':unittest.main()

import importlib.util, os, pathlib, unittest
import torch
PATH=pathlib.Path(__file__).resolve().parents[1] / 'runtime/python/sglang/srt/layers/quantization/private_draft_head.py'
class PrivateHeadTests(unittest.TestCase):
    def test_disabled_retains_exact_target(self):
        self.assertTrue(PATH.exists(), 'private draft head implementation is absent')
        spec=importlib.util.spec_from_file_location('trial_private_head',PATH)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        target=torch.nn.Linear(128,64,bias=False,dtype=torch.bfloat16)
        original=target.weight.detach().clone()
        os.environ.pop('SGLANG_PRIVATE_DRAFT_NVFP4_A16',None)
        self.assertIs(m.select_draft_head(target),target)
        self.assertTrue(torch.equal(original,target.weight))
    def test_private_shell_does_not_mutate_target_registries(self):
        spec=importlib.util.spec_from_file_location('trial_private_head',PATH)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertTrue(hasattr(m,'private_head_shell'), 'private ownership seam missing')
        target=torch.nn.Linear(128,64,bias=False,dtype=torch.bfloat16)
        target.shard_indices=('rank1',64,128)
        weight=target.weight; snapshot=weight.clone()
        private=m.private_head_shell(target)
        private.register_parameter('weight',torch.nn.Parameter(torch.zeros(64,64,dtype=torch.uint8),requires_grad=False))
        self.assertIsNot(private,target)
        self.assertIsNot(private._parameters,target._parameters)
        self.assertIs(target.weight,weight)
        self.assertTrue(torch.equal(target.weight,snapshot))
        self.assertEqual(private.shard_indices,target.shard_indices)

if __name__=='__main__':unittest.main()

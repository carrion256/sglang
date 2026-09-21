"""CPU-only dispatch checks; numerical kernel validation is a separate GPU gate."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get('QSA_SOURCE_ROOT', '/sgl-workspace/sglang'))
ATTN = ROOT / 'python/sglang/srt/layers/attention'


def load_sparse():
    spec = importlib.util.spec_from_file_location('paged_sparse_cpu', ATTN / 'qsa/sparse_attn.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Launch:
    def __getitem__(self, grid):
        self.grid = grid
        return self.call

    def call(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_wrapper_keeps_noncontiguous_pool_and_uses_cpu_lengths(dtype):
    mod = load_sparse()
    launch = Launch()
    mod._sparse_gqa_chunk_prefill = launch
    mod._get_best_config = lambda n: (16, 1, 2)
    # A materializing contiguous() here would copy the whole pool.
    k = torch.empty((64, 2, 256), dtype=dtype)[::2]
    v = torch.empty((64, 2, 256), dtype=dtype)[::2]
    q = torch.empty((5, 4, 256), dtype=torch.bfloat16)
    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    table = torch.zeros((4, 16), dtype=torch.int32)
    req = torch.tensor([3, 1, 0], dtype=torch.int64)
    out = mod.sparse_gqa_fwd_interface_triton_ck(
        q, k, v, torch.zeros((5, 8), dtype=torch.int32), cu, None,
        torch.tensor([10, 0, 12]), 0.0625,
        req_to_token=table, req_indices=req, max_q=3,
    )
    assert launch.args[1] is k and launch.args[2] is v
    assert launch.kwargs['PAGED'] is True
    assert launch.kwargs['req_to_token'] is table
    assert launch.kwargs['req_indices'] is req
    assert launch.grid == (3, 6)
    assert out.shape == q.shape


@pytest.mark.parametrize('mode', ['no_max', 'no_req', 'no_table', 'bad_table', 'bad_req'])
def test_wrapper_rejects_invalid_metadata(mode):
    mod = load_sparse()
    args = dict(req_to_token=torch.zeros((2, 16), dtype=torch.int32),
                req_indices=torch.tensor([1, 0]), max_q=2)
    if mode == 'no_max': args['max_q'] = None
    if mode == 'no_req': args['req_indices'] = None
    if mode == 'no_table': args['req_to_token'] = None
    if mode == 'bad_table': args['req_to_token'] = args['req_to_token'][:, ::2]
    if mode == 'bad_req': args['req_indices'] = torch.tensor([0])
    with pytest.raises(ValueError):
        mod.sparse_gqa_fwd_interface_triton_ck(
            torch.empty(4, 4, 256), torch.empty(32, 2, 256), torch.empty(32, 2, 256),
            torch.zeros(4, 8, dtype=torch.int32), torch.tensor([0, 2, 4]), None,
            torch.tensor([8, 8]), 0.0625, **args,
        )


def test_contiguous_reference_dispatch_retained():
    mod = load_sparse()
    launch = Launch()
    mod._sparse_gqa_chunk_prefill = launch
    mod._get_best_config = lambda n: (16, 1, 2)
    mod.sparse_gqa_fwd_interface_triton_ck(
        torch.empty(4, 4, 256), torch.empty(32, 2, 256), torch.empty(32, 2, 256),
        torch.zeros(4, 8, dtype=torch.int32), torch.tensor([0, 1, 4]),
        torch.tensor([0, 16, 32]), torch.tensor([16, 16]), 0.0625,
    )
    assert launch.kwargs['PAGED'] is False
    assert launch.grid == (3, 4)


class CudaShape(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize('path', ['prefix', 'no_prefix', 'speculative'])
def test_backend_routes_without_materializing_full_context(path):
    tree = ast.parse((ATTN / 'qwen_sparse_attn_backend.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenSparseAttnBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_extend')
    method.decorator_list = []
    paged, contiguous, spec = Mock(), Mock(), Mock()
    result = torch.zeros(2, 4, 256)
    for m in (paged, contiguous, spec): m.return_value = result
    ns = dict(torch=torch, F=F, Optional=__import__('typing').Optional,
              sparse_gqa_fwd_interface_triton_ck=paged,
              sparse_gqa_fwd_interface_triton=contiguous)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<backend-method>', 'exec'), ns)
    k_pool, v_pool = torch.empty(1000, 2, 256), torch.empty(1000, 2, 256)
    pool = SimpleNamespace(set_kv_buffer=Mock(), get_key_buffer=lambda _: k_pool,
                           get_value_buffer=lambda _: v_pool)
    table = torch.zeros((4, 1000), dtype=torch.int32)
    backend = SimpleNamespace(token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        _is_speculative_paged_mode=lambda _: path == 'speculative',
        _forward_paged_attention=spec, _pad_extend_output=lambda x, _: x)
    lengths = [2] if path == 'no_prefix' else [800]
    batch = SimpleNamespace(out_cache_loc=torch.tensor([8, 9]), forward_mode=None,
        extend_seq_lens_cpu=[2], seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor([2]), seq_lens=torch.tensor(lengths),
        req_pool_indices=torch.tensor([3]))
    layer = SimpleNamespace(tp_q_head_num=4, head_dim=256, layer_id=0, scaling=0.0625)
    q = torch.zeros(2, 4, 256).as_subclass(CudaShape)
    ns['forward_extend'](backend, q, torch.zeros(2, 2, 256), torch.zeros(2, 2, 256),
                         layer, batch, topk_indices=torch.zeros(2, 8, dtype=torch.int32))
    pool.set_kv_buffer.assert_called_once()
    selected = {'prefix': paged, 'no_prefix': contiguous, 'speculative': spec}[path]
    selected.assert_called_once()
    for other in (paged, contiguous, spec):
        if other is not selected: other.assert_not_called()
    if path == 'prefix':
        assert paged.call_args.args[1] is k_pool
        assert paged.call_args.args[2] is v_pool
        assert paged.call_args.kwargs['req_to_token'] is table
        assert paged.call_args.kwargs['max_q'] == 2

"""GPU acceptance gates. Run only in an explicitly authorized GPU test window."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

SOURCE = Path(os.environ.get('QSA_SOURCE_ROOT', '/sgl-workspace/sglang'))
spec = importlib.util.spec_from_file_location('paged_sparse_gpu', SOURCE / 'python/sglang/srt/layers/attention/qsa/sparse_attn.py')
sparse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')


def reference(q, k, v, table, requests, lengths, extensions, indices, scale):
    out = torch.zeros_like(q)
    row = 0
    for b, (length, extension) in enumerate(zip(lengths, extensions)):
        for j in range(extension):
            visible = length - extension + j + 1
            logical = indices[row]
            logical = logical[(logical >= 0) & (logical < visible)]
            slots = table[requests[b], logical.long()].long()
            slots = slots[(slots >= 0) & (slots < k.shape[0])]
            if slots.numel():
                keys = k[slots].to(q.dtype).float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                values = v[slots].to(q.dtype).float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
                scores = torch.einsum('hd,nhd->hn', q[row].float(), keys) * scale
                out[row] = torch.einsum('hn,nhd->hd', scores.softmax(-1), values).to(q.dtype)
            row += 1
    return out


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize('heads', [1, 2])
@pytest.mark.parametrize('extensions', [[1, 2, 3], [8, 0, 5]])
@pytest.mark.parametrize('invalid', [False, True])
def test_fragmented_prefixes_and_tails(dtype, heads, extensions, invalid):
    torch.manual_seed(47)
    dev = 'cuda'
    lengths = [129, 257, 511]
    rows = sum(extensions)
    q = torch.randn(rows, heads * 4, 256, dtype=torch.bfloat16, device=dev)
    # Noncontiguous pool strides test direct physical-slot addressing.
    k = torch.randn(4096, heads, 256, device=dev, dtype=torch.bfloat16).to(dtype)[::2]
    v = torch.randn(4096, heads, 256, device=dev, dtype=torch.bfloat16).to(dtype)[::2]
    table = torch.randperm(2048, device=dev).reshape(4, 512).int()
    requests = torch.tensor([3, 0, 2], device=dev, dtype=torch.int32)
    indices = torch.full((rows, 48), -1, device=dev, dtype=torch.int32)
    row = 0
    for n, ext in zip(lengths, extensions):
        for j in range(ext):
            visible = n - ext + j + 1
            # Unique, deliberately unsorted positions, including the newest token.
            chosen = torch.randperm(visible - 1, device=dev)[:47]
            indices[row] = torch.cat([chosen, torch.tensor([visible - 1], device=dev)]).int()
            row += 1
    if invalid:
        indices[0] = -1
        indices[1, :16] = -1  # Entire initial tile empty; later tiles still valid.
        indices[-1, 0] = 511  # Future position or out of sequence range.
        indices[-1, 1] = 9999
    cu = torch.tensor([0] + list(__import__('itertools').accumulate(extensions)), device=dev, dtype=torch.int32)
    seq = torch.tensor(lengths, device=dev, dtype=torch.int32)
    out = sparse.sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu, None, seq, 256 ** -0.5,
        req_to_token=table, req_indices=requests, max_q=max(extensions))
    expected = reference(q, k, v, table, requests, lengths, extensions, indices, 256 ** -0.5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.025, rtol=0.025)
    if not invalid:
        kp = [k[table[requests[b], :n].long()].to(q.dtype) for b, n in enumerate(lengths)]
        vp = [v[table[requests[b], :n].long()].to(q.dtype) for b, n in enumerate(lengths)]
        cuk = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)), device=dev, dtype=torch.int32)
        old = sparse.sparse_gqa_fwd_interface_triton_ck(q, torch.cat(kp), torch.cat(vp), indices, cu, cuk, seq, 256 ** -0.5)
        torch.testing.assert_close(out, old, atol=0.01, rtol=0.01)


def test_512k_prefix_temporary_memory_is_bounded():
    """Isolated allocator check, not a substitute for full-engine qualification."""
    n, heads, dim, rows = 524288, 2, 256, 8
    k = torch.zeros(n, heads, dim, dtype=torch.float8_e4m3fn, device='cuda')
    v = torch.zeros_like(k)
    q = torch.ones(rows, heads * 4, dim, dtype=torch.bfloat16, device='cuda')
    table = torch.arange(n - 1, -1, -1, device='cuda', dtype=torch.int32)[None]
    req = torch.tensor([0], device='cuda', dtype=torch.int32)
    cu = torch.tensor([0, rows], device='cuda', dtype=torch.int32)
    lens = torch.tensor([n], device='cuda', dtype=torch.int32)
    idx = torch.arange(48, device='cuda', dtype=torch.int32).expand(rows, -1).contiguous()
    def paged():
        return sparse.sparse_gqa_fwd_interface_triton_ck(q, k, v, idx, cu, None, lens, dim ** -0.5,
            req_to_token=table, req_indices=req, max_q=rows)
    paged()  # Compile before allocator measurement.
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = paged()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    assert extra < 16 * 1024 * 1024, extra
    assert torch.count_nonzero(out) == 0


def test_physical_offsets_above_int32():
    """Requires about 4.1 GiB for FP8 pools; catches slot*head*dim overflow."""
    heads, dim = 2, 256
    slot = 2**31 // (heads * dim) + 64
    k = torch.empty(slot + 1, heads, dim, dtype=torch.float8_e4m3fn, device='cuda')
    v = torch.empty_like(k)
    k[slot].fill_(1)
    v[slot].fill_(2)
    q = torch.ones(1, 8, dim, dtype=torch.bfloat16, device='cuda')
    out = sparse.sparse_gqa_fwd_interface_triton_ck(
        q, k, v, torch.tensor([[0]], device='cuda', dtype=torch.int32),
        torch.tensor([0, 1], device='cuda', dtype=torch.int32), None,
        torch.tensor([1], device='cuda', dtype=torch.int32), dim ** -0.5,
        req_to_token=torch.tensor([[slot]], device='cuda', dtype=torch.int32),
        req_indices=torch.tensor([0], device='cuda', dtype=torch.int32), max_q=1)
    torch.testing.assert_close(out, torch.full_like(out, 2))

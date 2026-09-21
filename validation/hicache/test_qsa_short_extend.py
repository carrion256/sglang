"""Partial compression groups must not gather beyond a short cached extension."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8])
def test_short_extend_gather_bounds(rows):
    ratio = 4
    prefix = 64
    writes, ends, _, members = QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=torch.arange(128).reshape(1, -1),
        start_blocks=torch.tensor([prefix // ratio]),
        end_blocks=torch.tensor([(prefix + rows) // ratio]),
        capacity=rows // ratio + 1,
        compress_ratio=ratio,
        row_token_starts=torch.tensor([0]),
        prefix_lens=torch.tensor([prefix]),
    )
    stored = []
    pool = SimpleNamespace(set_qsa_compressed_k_buffer=lambda layer, slots, keys: stored.append(keys))
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        compress_member_rows=members,
        is_cuda_graph=False,
        write_locs=writes,
        compress_group_positions=ends,
        extend_rope_matrix=torch.zeros((rows, 3), dtype=torch.int64),
    )
    indexer = SimpleNamespace(
        compress_ratio=ratio,
        layer_id=0,
        _use_fused_compress=lambda pool: False,
        _rope_from_matrix=lambda values: values,
        normalize_compressed_keys=lambda keys, positions: keys,
    )
    keys = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 1, 8)
    QSAIndexer.update_key_state_and_compress(
        indexer, keys, torch.arange(prefix, prefix + rows),
        torch.zeros((3, rows), dtype=torch.int64), metadata, state_stored=True,
    )
    assert len(stored) == 1
    valid = writes != 0
    expected = keys[: rows // ratio * ratio].reshape(-1, ratio, 1, 8).mean(1)
    torch.testing.assert_close(stored[0][valid], expected)

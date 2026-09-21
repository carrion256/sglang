"""Common KV/recurrent/indexer checkpoint boundaries."""
from itertools import permutations
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, PoolHitPolicy, PoolName, PoolTransfer
from test_hicache_file_local import storage

def oracle(keys, inventory, transfers):
    valid = []
    for n in range(1, len(keys) + 1):
        if not all((k, None) in inventory for k in keys[:n]):
            continue
        if all(all((k, t.name) in inventory for k in (
            keys[:n] if t.hit_policy == PoolHitPolicy.ALL_PAGES else
            keys[max(0, n - max(1, len(t.keys or []))):n]
        )) for t in transfers):
            valid.append(n)
    return max(valid, default=0)


def test_common_boundary_exhaustive():
    keys = ['a', 'b', 'c', 'd']
    for tail_length in (1, 2, 5):
        transfers = [PoolTransfer(PoolName.MAMBA, keys=['x'] * tail_length,
                                  hit_policy=PoolHitPolicy.TRAILING_PAGES),
                     PoolTransfer(PoolName.QSA_INDEXER, hit_policy=PoolHitPolicy.ALL_PAGES)]
        for m in range(16):
            for q in range(16):
                inventory = {(k, None) for k in keys}
                inventory |= {(k, PoolName.MAMBA) for i, k in enumerate(keys) if m >> i & 1}
                inventory |= {(k, PoolName.QSA_INDEXER) for i, k in enumerate(keys) if q >> i & 1}
                backend = SimpleNamespace(
                    _get_component_key=lambda k, n=None: f'{k}:{n}',
                    _collect_existing_component_keys=lambda k, t: {f'{a}:{b}.bin' for a, b in inventory},
                )
                for order in permutations(transfers):
                    result = HiCacheFile.batch_exists_v2(backend, keys, order)
                    assert result.kv_hit_pages == oracle(keys, inventory, order)


@pytest.mark.parametrize('metadata', [False, True])
@pytest.mark.parametrize('rank', [0, 1])
def test_real_file_common_checkpoint(tmp_path, metadata, rank):
    backend = storage(tmp_path, rank=rank, metadata=metadata)
    keys = ['a', 'b', 'c', 'd']
    data = torch.tensor([17], dtype=torch.int32)
    for key in keys:
        assert backend.set(key, data)
    for key in ['b', 'd']:
        assert backend.set(key + '.mamba', data)
    for key in keys[:3]:
        assert backend.set(key + '.qsa_indexer', data)
    transfers = [PoolTransfer(PoolName.MAMBA, keys=['d'], hit_policy=PoolHitPolicy.TRAILING_PAGES),
                 PoolTransfer(PoolName.QSA_INDEXER, hit_policy=PoolHitPolicy.ALL_PAGES)]
    for order in permutations(transfers):
        result = backend.batch_exists_v2(keys, order)
        assert result.kv_hit_pages == 2
        restored = backend.get(keys[result.kv_hit_pages - 1] + '.mamba', torch.empty_like(data))
        assert torch.equal(restored, data)


def test_multiple_trailing_pools_and_kv_gaps():
    keys = ['a', 'b', 'c']
    transfers = [PoolTransfer(PoolName.MAMBA, keys=['c'], hit_policy=PoolHitPolicy.TRAILING_PAGES),
                 PoolTransfer(PoolName.DRAFT, keys=['b', 'c'], hit_policy=PoolHitPolicy.TRAILING_PAGES),
                 PoolTransfer(PoolName.QSA_INDEXER, hit_policy=PoolHitPolicy.ALL_PAGES)]
    for m in range(8):
        for d in range(8):
            for kv in range(4):
                inventory = {(k, None) for k in keys[:kv]}
                inventory |= {(k, PoolName.MAMBA) for i, k in enumerate(keys) if m >> i & 1}
                inventory |= {(k, PoolName.DRAFT) for i, k in enumerate(keys) if d >> i & 1}
                inventory |= {(k, PoolName.QSA_INDEXER) for k in keys}
                backend = SimpleNamespace(
                    _get_component_key=lambda k, n=None: f'{k}:{n}',
                    _collect_existing_component_keys=lambda k, t: {f'{a}:{b}.bin' for a, b in inventory},
                )
                for order in permutations(transfers):
                    assert HiCacheFile.batch_exists_v2(backend, keys, order).kv_hit_pages == oracle(keys, inventory, order)
                assert HiCacheFile.batch_exists_v2(backend, keys, []).kv_hit_pages == kv
                assert HiCacheFile.batch_exists_v2(backend, [], []).kv_hit_pages == 0

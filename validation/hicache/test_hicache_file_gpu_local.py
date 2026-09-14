"""Real Qwen-shaped GPU↔RAM↔file checkpoints, including packed MTP and PLE."""

import os

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from test_hicache_file_local import storage
from test_hicache_ple_gpu_local import exercise_async_checkpoint


def test_gpu_disk_checkpoint_survives_backend_reconstruction(tmp_path):
    rank = int(os.environ.get("QWEN_HICACHE_TP_RANK", "0"))

    def roundtrip(kv_host, state_host, kv_rows, state_rows, epoch):
        def reopen():
            backend = storage(tmp_path, rank)
            backend.register_mem_host_pool_v2(kv_host, PoolName.KV)
            backend.register_mem_host_pool_v2(state_host, PoolName.MAMBA)
            return backend

        transfers = [
            PoolTransfer(PoolName.KV, host_indices=kv_rows, keys=[str(epoch)]),
            PoolTransfer(PoolName.MAMBA, host_indices=state_rows, keys=[str(epoch)]),
        ]
        assert reopen().batch_set_v2(transfers) == {
            PoolName.KV: [True],
            PoolName.MAMBA: [True],
        }
        kv_host.kv_buffer.zero_()
        for tensor in state_host.get_hybrid_pool_buffer():
            tensor.zero_()
        backend = reopen()
        expected_bytes = (epoch + 1) * (
            64 * kv_host.size_per_token + state_host.size_per_token
        )
        assert backend._evictor._total_bytes == expected_bytes
        assert backend._evictor._total_bytes <= backend._evictor.max_size_bytes
        assert backend.batch_get_v2(transfers) == {
            PoolName.KV: [True],
            PoolName.MAMBA: [True],
        }

    exercise_async_checkpoint(disk_roundtrip=roundtrip)

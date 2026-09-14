"""Real target/draft index copies with relocation, file restart and layer waits."""

import os

import pytest
import torch

from sglang.srt.managers.cache_controller import CacheOperation, LayerDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.qsa_pool_host import QSAPagedHostPool

from test_hicache_file_local import storage
from test_hicache_ple_gpu_local import make_kv
from test_hicache_qsa_local import make_index_pool


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
@pytest.mark.parametrize("on_disk", [False, True])
def test_async_qsa_target_and_draft_relocation(tmp_path, layout, on_disk):
    assert torch.cuda.is_available(), "Real CUDA is required for this gate"
    main, draft = make_kv(12), make_kv(1)
    qsa, draft_qsa = make_index_pool(12, "cuda"), make_index_pool(1, "cuda")
    made = []
    try:
        kv_host = MHATokenToKVPoolHost(
            main, 2, 0, 64, layout, mtp_draft_device_pools=(draft,)
        )
        made.append(kv_host)
        index_host = QSAPagedHostPool((qsa, draft_qsa), kv_host.size, 64, layout)
        made.append(index_host)
        kv_map = {layer: i for i, layer in enumerate(range(3, 48, 4))}
        kv_map[48] = 12
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(PoolName.KV, kv_host, main, kv_map.get, True),
                PoolEntry(PoolName.QSA_INDEXER, index_host, qsa, kv_map.get),
            ]
        )
        controller.has_draft, controller.has_mtp_draft = False, True
        controller.mtp_draft_device_pools = (draft,)
        controller.layer_num, controller.io_backend, controller.device = (
            48,
            "kernel",
            "cuda",
        )
        counter, engine = LayerDoneCounter(48), L2TransferEngine("kernel")
        qsa.layer_transfer_counter = counter
        rows = kv_host.alloc(128)
        source, destination = torch.arange(64, 192, device="cuda"), torch.arange(
            128, 256, device="cuda"
        )
        source_index, dest_index = torch.arange(16, 48, device="cuda"), torch.arange(
            32, 64, device="cuda"
        )
        all_kv = main.k_buffer + main.v_buffer + draft.k_buffer + draft.v_buffer
        all_index = (
            qsa.qsa_compressed_k_buffer_pool + draft_qsa.qsa_compressed_k_buffer_pool
        )
        assert index_host.layer_num == 13 and index_host.size_per_token == 13 * 64
        for epoch in range(4):
            for i, buffer in enumerate(all_kv):
                buffer[source] = (
                    torch.arange(128, device="cuda")[:, None, None] + i * 7 + epoch
                ).to(torch.uint8)
            for i, buffer in enumerate(all_index):
                buffer[source_index] = (
                    torch.arange(32 * 128, device="cuda").reshape(32, 1, 128) % 191
                    + i
                    + epoch
                ).to(torch.bfloat16)
            expected_kv = [x[source].clone() for x in all_kv]
            expected_index = [x[source_index].clone() for x in all_index]
            write = CacheOperation(
                rows,
                source,
                epoch,
                pool_transfers=[
                    PoolTransfer(
                        PoolName.QSA_INDEXER,
                        rows,
                        source,
                        indices_from_pool=PoolName.KV,
                    )
                ],
            )
            assert controller._transfer_num_bytes(write) == 128 * (
                kv_host.size_per_token + index_host.size_per_token
            )
            copied = engine.submit_device_to_host(
                controller._l2_transfers(*controller._move_write_operation(write))
            )
            copied.finish_event.synchronize()
            if on_disk:

                def reopen():
                    backend = storage(
                        tmp_path, int(os.environ.get("QWEN_HICACHE_TP_RANK", "0"))
                    )
                    backend.register_mem_host_pool_v2(kv_host, PoolName.KV)
                    backend.register_mem_host_pool_v2(index_host, PoolName.QSA_INDEXER)
                    return backend

                keys = [f"{epoch}-{i}" for i in range(2)]
                transfers = [
                    PoolTransfer(name, rows, keys=keys)
                    for name in (PoolName.KV, PoolName.QSA_INDEXER)
                ]
                expected_hits = {
                    name: [True, True] for name in (PoolName.KV, PoolName.QSA_INDEXER)
                }
                assert reopen().batch_set_v2(transfers) == expected_hits
                kv_host.kv_buffer.zero_()
                for buffer in index_host.get_hybrid_pool_buffer():
                    buffer.zero_()
                assert reopen().batch_get_v2(transfers) == expected_hits
            for buffer in all_kv:
                buffer[destination] = 255
            for buffer in all_index:
                buffer[dest_index] = -200
            load = CacheOperation(
                rows,
                destination,
                epoch,
                pool_transfers=[
                    PoolTransfer(
                        PoolName.QSA_INDEXER,
                        rows,
                        destination,
                        indices_from_pool=PoolName.KV,
                    )
                ],
            )
            transfers = controller._l2_load_transfers(
                *controller.move_hybrid_indices(load)
            )
            assert len(transfers) == 4 and sum(x.is_draft for x in transfers) == 2
            event_id = counter.update_producer()
            counter.set_consumer(event_id)
            with torch.cuda.stream(engine.host_to_device_stream):
                torch.cuda._sleep(20_000_000)
            restored = engine.submit_host_to_device(
                transfers, layer_num=48, on_layer_done=counter.events[event_id].complete
            )
            # This read must wait for the index copy, before a full-transfer sync.
            early = qsa.get_qsa_compressed_k_buffer(3)[dest_index].clone()
            torch.cuda.current_stream().synchronize()
            assert torch.equal(early, expected_index[0])
            restored.finish_event.synchronize()
            for actual, expected in zip(all_kv, expected_kv):
                assert torch.equal(actual[destination], expected)
            for actual, expected in zip(all_index, expected_index):
                assert torch.equal(actual[dest_index], expected)
    finally:
        torch.cuda.synchronize()
        for host in made:
            host.destroy()

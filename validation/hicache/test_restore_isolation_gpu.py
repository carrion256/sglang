"""Distinct hybrid checkpoints restored out of order through reused physical slots."""

import os

import pytest
import torch

from sglang.srt.managers.cache_controller import CacheOperation, LayerDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, MambaPoolHost, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.qsa_pool_host import QSAPagedHostPool

from test_hicache_file_local import storage
from test_hicache_ple_gpu_local import make_kv, make_qwen_state
from test_hicache_ple_local import snapshot, tensors, assert_state
from test_hicache_qsa_local import make_index_pool


@pytest.mark.parametrize("metadata", [False, True])
def test_out_of_order_complete_hybrid_restore(tmp_path, metadata):
    assert torch.cuda.is_available()
    rank = int(os.environ.get("QWEN_HICACHE_TP_RANK", "0"))
    main, draft, recurrent = make_kv(12), make_kv(1), make_qwen_state()
    qsa, draft_qsa = make_index_pool(12, "cuda"), make_index_pool(1, "cuda")
    hosts = []
    try:
        kv_host = MHATokenToKVPoolHost(main, 2, 0, 64, "page_first", mtp_draft_device_pools=(draft,))
        hosts.append(kv_host)
        state_host = MambaPoolHost(recurrent, 2, 0, layout="page_first")
        hosts.append(state_host)
        index_host = QSAPagedHostPool((qsa, draft_qsa), kv_host.size, 64, "page_first")
        hosts.append(index_host)
        kv_map = {layer: i for i, layer in enumerate(range(3, 48, 4))}
        kv_map[48] = 12
        state_map = {layer: i for i, layer in enumerate(recurrent.mamba_layer_ids)}
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([
            PoolEntry(PoolName.KV, kv_host, main, kv_map.get, True),
            PoolEntry(PoolName.MAMBA, state_host, recurrent, state_map.get),
            PoolEntry(PoolName.QSA_INDEXER, index_host, qsa, kv_map.get),
        ])
        controller.has_draft, controller.has_mtp_draft = False, True
        controller.mtp_draft_device_pools = (draft,)
        controller.layer_num, controller.io_backend, controller.device = 48, "kernel", "cuda"
        counter, engine = LayerDoneCounter(48), L2TransferEngine("kernel")
        rows, state_rows = kv_host.alloc(64), state_host.alloc(1)
        source = torch.arange(64, 128, device="cuda")
        state_source = torch.tensor([1], device="cuda")
        all_kv = main.k_buffer + main.v_buffer + draft.k_buffer + draft.v_buffer
        all_index = qsa.qsa_compressed_k_buffer_pool + draft_qsa.qsa_compressed_k_buffer_pool
        expected = {}

        def backend():
            result = storage(tmp_path, rank=rank, cap=1_000_000_000, metadata=metadata)
            for name, host in ((PoolName.KV, kv_host), (PoolName.MAMBA, state_host), (PoolName.QSA_INDEXER, index_host)):
                result.register_mem_host_pool_v2(host, name)
            return result

        def disk_transfers(checkpoint):
            key = f"conversation-{checkpoint}-checkpoint"
            return [PoolTransfer(name, host_indices=indices, keys=[key]) for name, indices in
                    ((PoolName.KV, rows), (PoolName.MAMBA, state_rows), (PoolName.QSA_INDEXER, rows))]

        def operation(device_rows, state_slot, checkpoint):
            return CacheOperation(rows, device_rows, checkpoint, pool_transfers=[
                PoolTransfer(PoolName.MAMBA, state_rows, state_slot),
                PoolTransfer(PoolName.QSA_INDEXER, rows, device_rows, indices_from_pool=PoolName.KV),
            ])

        expected_hits = {name: [True] for name in (PoolName.KV, PoolName.MAMBA, PoolName.QSA_INDEXER)}
        # Each write overwrites exactly the same GPU and host locations.
        for checkpoint in range(5):
            for i, value in enumerate(all_kv):
                value[source] = (torch.arange(64, device="cuda")[:, None, None] + i * 7 + checkpoint * 31 + rank * 13).to(torch.uint8)
            for i, value in enumerate(all_index):
                value[16:32] = (torch.arange(16 * 128, device="cuda").reshape(16, 1, 128) % 97 + i + checkpoint * 131 + rank * 17).to(torch.bfloat16)
            for i, value in enumerate(tensors(recurrent).values()):
                value[:, state_source] = 21 + i + checkpoint * 17 + rank * 3
            expected[checkpoint] = ([v[source].clone() for v in all_kv], [v[16:32].clone() for v in all_index], snapshot(recurrent, state_source))
            written = engine.submit_device_to_host(controller._l2_transfers(*controller._move_write_operation(operation(source, state_source, checkpoint))))
            written.finish_event.synchronize()
            assert backend().batch_set_v2(disk_transfers(checkpoint)) == expected_hits

        # A different logical conversation must not inherit the last staging copy.
        absent = backend()
        missing = disk_transfers("never-written")
        assert absent.batch_exists_v2(["conversation-never-written-checkpoint"], missing).kv_hit_pages == 0
        assert absent.batch_get_v2(missing) == {name: [False] for name in expected_hits}

        for iteration, checkpoint in enumerate((4, 0, 3, 1, 2, 0, 4, 1, 3, 2)):
            # Alternate destinations and wrap the layer-event ring repeatedly.
            start = 128 if iteration % 2 else 192
            destination = torch.arange(start, start + 64, device="cuda")
            state_dest = torch.tensor([4 + iteration % 2], device="cuda")
            index_dest = slice(start // 4, (start + 64) // 4)
            kv_host.kv_buffer.zero_()
            for host in (state_host, index_host):
                for value in host.get_hybrid_pool_buffer():
                    value.zero_()
            for value in all_kv:
                value[destination] = 255
            for value in all_index:
                value[index_dest] = -200
            for value in tensors(recurrent).values():
                value[:, state_dest] = -200
            assert backend().batch_get_v2(disk_transfers(checkpoint)) == expected_hits
            op = operation(destination, state_dest, iteration)
            transfers = controller._l2_load_transfers(*controller.move_hybrid_indices(op))
            assert len(transfers) == 5
            event_id = counter.update_producer()
            counter.set_consumer(event_id)
            with torch.cuda.stream(engine.host_to_device_stream):
                torch.cuda._sleep(2_000_000)
            restored = engine.submit_host_to_device(transfers, layer_num=48, on_layer_done=counter.events[event_id].complete)
            restored.finish_event.synchronize()
            kv_expected, index_expected, state_expected = expected[checkpoint]
            for actual, want in zip(all_kv, kv_expected):
                assert torch.equal(actual[destination], want)
            for actual, want in zip(all_index, index_expected):
                assert torch.equal(actual[index_dest], want)
            assert_state(recurrent, state_dest, state_expected)
    finally:
        torch.cuda.synchronize()
        for host in hosts:
            host.destroy()

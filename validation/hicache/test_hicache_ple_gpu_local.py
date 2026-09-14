"""CUDA-only integration checks with Qwen TP2 state shapes and real transfers."""

import torch

from sglang.srt.managers.cache_controller import CacheOperation, LayerDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MambaPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

from test_hicache_ple_local import assert_state, poison, snapshot, tensors


def make_kv(layer_num):
    # Synthetic values, but the serving pool's exact FP8 byte layout and geometry.
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.size, pool.page_size = 256, 64
    pool.dtype, pool.store_dtype = torch.float8_e4m3fn, torch.uint8
    pool.layer_num, pool.start_layer, pool.end_layer = layer_num, 0, layer_num
    pool.head_num, pool.head_dim, pool.device = 1, 256, "cuda"
    pool.k_buffer = [
        torch.zeros((320, 1, 256), dtype=torch.uint8, device="cuda")
        for _ in range(layer_num)
    ]
    pool.v_buffer = [torch.zeros_like(x) for x in pool.k_buffer]
    pool.k_data_ptrs = torch.tensor(
        [x.data_ptr() for x in pool.k_buffer], dtype=torch.uint64, device="cuda"
    )
    pool.v_data_ptrs = torch.tensor(
        [x.data_ptr() for x in pool.v_buffer], dtype=torch.uint64, device="cuda"
    )
    return pool


def make_qwen_state():
    pool = MambaPool.__new__(MambaPool)
    pool.size, pool.num_mamba_layers, pool.device = 7, 36, "cuda"
    pool.mamba_layer_ids = [layer for layer in range(48) if layer % 4 != 3]
    pool.mamba_cache = MambaPool.State(
        conv=[torch.zeros((36, 8, 5120, 3), dtype=torch.bfloat16, device="cuda")],
        temporal=torch.zeros(
            (36, 8, 24, 128, 128), dtype=torch.bfloat16, device="cuda"
        ),
    )
    conv = ShortConvPool.__new__(ShortConvPool)
    conv.conv_state = torch.zeros((1, 8, 10240, 9), dtype=torch.bfloat16, device="cuda")
    conv.layer_map = {2: 0}
    ngram = NGramPool.__new__(NGramPool)
    ngram.context = torch.zeros((8, 2), dtype=torch.int64, device="cuda")
    pool._slot_siblings, pool.replayssm_cache_base = [conv, ngram], None
    return pool


def exercise_async_checkpoint(disk_roundtrip=None):
    assert torch.cuda.is_available(), "This integration check requires real CUDA"
    main, draft, recurrent = make_kv(12), make_kv(1), make_qwen_state()
    made = []
    try:
        kv_host = MHATokenToKVPoolHost(
            main, 2, 0, 64, "page_first", mtp_draft_device_pools=(draft,)
        )
        made.append(kv_host)
        state_host = MambaPoolHost(recurrent, 2, 0, layout="page_first")
        made.append(state_host)
        kv_map = {layer: i for i, layer in enumerate(range(3, 48, 4))}
        kv_map[48] = 12
        state_map = {layer: i for i, layer in enumerate(recurrent.mamba_layer_ids)}
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(PoolName.KV, kv_host, main, kv_map.get, True),
                PoolEntry(PoolName.MAMBA, state_host, recurrent, state_map.get),
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
        request_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        request_pool.start_layer, request_pool.mamba_map = 0, state_map
        request_pool.layer_transfer_counter = counter
        request_pool.short_conv_pool, request_pool.ngram_pool = recurrent._slot_siblings
        kv_rows, state_rows = kv_host.alloc(64), state_host.alloc(1)
        kv_src, kv_dst = torch.arange(64, device="cuda"), torch.arange(
            128, 192, device="cuda"
        )
        state_src, state_dst = torch.tensor([1], device="cuda"), torch.tensor(
            [4, 5], device="cuda"
        )
        all_kv = main.k_buffer + main.v_buffer + draft.k_buffer + draft.v_buffer
        assert kv_host.layer_num == 13
        assert kv_host.size_per_token == 13 * 2 * 256
        assert (
            state_host.size_per_token
            == 36 * (5120 * 3 + 24 * 128 * 128) * 2 + 10240 * 9 * 2 + 2 * 8
        )
        assert (
            sum(
                t.numel() * t.element_size()
                for t in state_host.get_hybrid_pool_buffer()
            )
            == state_host.size * state_host.size_per_token
        )

        # Reuse slots and wrap the actual three-event producer/consumer ring.
        for epoch in range(4):
            for i, tensor in enumerate(all_kv):
                tensor[kv_src] = (
                    torch.arange(64, device="cuda")[:, None, None] + 7 * i + epoch
                ).to(torch.uint8)
            for i, tensor in enumerate(tensors(recurrent).values()):
                tensor[:, state_src] = epoch + i + 21
            expected_kv = [t[kv_src].clone() for t in all_kv]
            expected_state = {
                k: v.repeat(1, 2, *([1] * (v.ndim - 2)))
                for k, v in snapshot(recurrent, state_src).items()
            }
            write_op = CacheOperation(
                kv_rows,
                kv_src,
                epoch,
                pool_transfers=[PoolTransfer(PoolName.MAMBA, state_rows, state_src)],
            )
            write_args = controller._move_write_operation(write_op)
            written = engine.submit_device_to_host(
                controller._l2_transfers(*write_args)
            )
            written.finish_event.synchronize()
            if disk_roundtrip is not None:
                disk_roundtrip(kv_host, state_host, kv_rows, state_rows, epoch)
            for tensor in all_kv:
                tensor[kv_dst] = 255
            poison(recurrent, state_dst)

            load_op = CacheOperation(
                kv_rows,
                kv_dst,
                epoch,
                pool_transfers=[
                    PoolTransfer(PoolName.MAMBA, state_rows.repeat(2), state_dst)
                ],
            )
            load_args = controller.move_hybrid_indices(load_op)
            transfers = controller._l2_load_transfers(*load_args)
            assert len(transfers) == 3 and transfers[-1].is_draft
            event_index = counter.update_producer()
            counter.set_consumer(event_index)
            with torch.cuda.stream(engine.host_to_device_stream):
                torch.cuda._sleep(20_000_000)
            restored = engine.submit_host_to_device(
                transfers,
                layer_num=48,
                on_layer_done=counter.events[event_index].complete,
            )

            # Capture PLE reads before any host-side synchronization of the restore.
            early_ngram = request_pool.get_ngram_context(state_dst).clone()
            early_conv = request_pool.short_conv_layer_cache(2)[state_dst].clone()
            torch.cuda.current_stream().synchronize()
            assert torch.equal(early_ngram, expected_state["ple_ngram"][0])
            assert torch.equal(early_conv, expected_state["ple_conv"][0])
            restored.finish_event.synchronize()
            assert_state(recurrent, state_dst, expected_state)
            for actual, expected in zip(all_kv, expected_kv):
                assert torch.equal(actual[kv_dst], expected)
            assert kv_host.available_size() == kv_host.size - 64
            assert state_host.available_size() == state_host.size - 1
    finally:
        torch.cuda.synchronize()
        for host in made:
            host.destroy()


def test_async_qwen_main_draft_mamba_and_ple_checkpoint():
    exercise_async_checkpoint()

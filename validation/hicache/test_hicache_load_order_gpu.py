"""Exercise delayed GPU producers through the restored-state consumer wait."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController, LayerDoneCounter
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner


def test_deferred_copy_sees_all_restored_layers():
    assert torch.cuda.is_available()
    source = torch.zeros((3, 1024), device="cuda")
    destination = torch.zeros_like(source)
    src_index = torch.tensor([0], device="cuda")
    dst_index = torch.tensor([1], device="cuda")
    restore_stream = torch.cuda.Stream()
    restore_stream.wait_stream(torch.cuda.current_stream())
    counter = LayerDoneCounter(3)
    index = counter.update_producer()
    counter.set_consumer(index)
    with torch.cuda.stream(restore_stream):
        for layer in range(3):
            torch.cuda._sleep(20_000_000)
            source[layer].fill_(layer + 17)
            counter.events[index].complete(layer)

    pool = object.__new__(HybridReqToTokenPool)
    pool.layer_transfer_counter = counter
    pool.mamba_ckpt_pool = None
    pool.translate_mamba_indices = lambda indices: indices
    pool.mamba_pool = SimpleNamespace(copy_from=lambda *args: destination.copy_(source))
    runner = object.__new__(ModelRunner)
    runner.req_to_token_pool = pool
    runner.is_draft_worker = False
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mamba_clear_indices=None,
        mamba_cow_src_indices=src_index,
        mamba_cow_dst_indices=dst_index,
    )
    runner._maybe_execute_deferred_mamba_cow_and_clear(batch)
    actual = destination.cpu()
    restore_stream.synchronize()
    expected = torch.arange(17, 20, dtype=actual.dtype)[:, None].expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_restore_follows_inflight_write_to_reclaimed_page():
    assert torch.cuda.is_available()
    destination = torch.zeros((1, 1024), device="cuda")
    host = torch.full((1, 1024), 17.0, pin_memory=True)
    indices = torch.tensor([0])
    forward_stream = torch.cuda.Stream()
    forward_stream.wait_stream(torch.cuda.current_stream())
    engine = L2TransferEngine("direct")
    counter = LayerDoneCounter(1)
    host_pool = SimpleNamespace(
        layer_num=1,
        load_to_device_per_layer=lambda *args, **kwargs: destination.copy_(host, non_blocking=True),
    )
    transfer = L2Transfer(host_pool, None, indices, indices)
    controller = object.__new__(HiCacheController)
    op = CacheOperation(indices, indices, 42)
    controller.load_queue = [op]
    controller.ack_load_queue = []
    controller.layer_num = 1
    controller.layer_done_counter = counter
    controller.load_fence_stream = forward_stream
    controller.l2_transfer_engine = engine
    controller._move_op_indices = lambda op: (indices, indices, [])
    controller._l2_load_transfers = lambda *args: [transfer]
    controller._num_tokens_by_pool = lambda op: {}
    controller._transfer_num_bytes = lambda op: host.numel() * host.element_size()
    with torch.cuda.stream(forward_stream):
        torch.cuda._sleep(200_000_000)
        destination.fill_(99)
    controller.start_loading()
    controller.ack_load_queue[0].finish_event.synchronize()
    forward_stream.synchronize()
    torch.testing.assert_close(destination.cpu(), host, rtol=0, atol=0)

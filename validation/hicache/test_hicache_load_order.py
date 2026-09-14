"""Restore ordering for reclaimed pages and whole-slot recurrent-state copies."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner


@pytest.mark.parametrize("fenced", [False, True])
def test_restore_fences_before_transfer_submission(fenced):
    calls = []
    controller = object.__new__(HiCacheController)
    op = CacheOperation(torch.tensor([0]), torch.tensor([1]), 42)
    controller.load_queue = [op]
    controller.ack_load_queue = []
    controller.layer_num = 3
    event = SimpleNamespace(start_event=Mock(), complete=Mock())
    controller.layer_done_counter = SimpleNamespace(
        update_producer=lambda: 0, events=[event]
    )
    controller.load_fence_stream = object() if fenced else None
    controller._move_op_indices = lambda op: (op.host_indices, op.device_indices, [])
    controller._l2_load_transfers = lambda *args: []
    controller._num_tokens_by_pool = lambda op: {}
    controller._transfer_num_bytes = lambda op: 0

    def wait(stream):
        assert stream is controller.load_fence_stream
        calls.append("wait")

    def submit(*args, **kwargs):
        calls.append("submit")
        return SimpleNamespace(start_event=object(), finish_event=object(), timing_enabled=False)

    controller.l2_transfer_engine = SimpleNamespace(
        host_to_device_stream=SimpleNamespace(wait_stream=wait),
        submit_host_to_device=submit,
    )
    assert controller.start_loading() == 0
    assert calls == (["wait", "submit"] if fenced else ["submit"])
    assert controller.start_loading() == -1
    assert len(controller.ack_load_queue) == 1


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("has_cache", [False, True])
def test_whole_slot_copy_waits_for_final_layer(checkpoint, has_cache):
    calls = []
    pool = object.__new__(HybridReqToTokenPool)
    pool.layer_transfer_counter = (
        SimpleNamespace(num_layers=48, wait_until=lambda i: calls.append(("wait", i)))
        if has_cache else None
    )
    pool.translate_mamba_indices = lambda indices: indices
    pool.mamba_pool = SimpleNamespace(copy_from=lambda *args: calls.append(("copy",)))
    pool.mamba_ckpt_pool = (
        SimpleNamespace(load_to_active=lambda *args: calls.append(("copy",)))
        if checkpoint else None
    )
    runner = object.__new__(ModelRunner)
    runner.req_to_token_pool = pool
    runner.is_draft_worker = False
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mamba_clear_indices=None,
        mamba_cow_src_indices=torch.tensor([2]),
        mamba_cow_dst_indices=torch.tensor([5]),
    )
    runner._maybe_execute_deferred_mamba_cow_and_clear(batch)
    assert calls == ([("wait", 47), ("copy",)] if has_cache else [("copy",)])
    assert batch.mamba_cow_src_indices is None
    runner._maybe_execute_deferred_mamba_cow_and_clear(batch)
    assert calls.count(("copy",)) == 1

"""Real file-backend isolation with reused buffers and rank-local namespaces."""

import pytest
import torch

from test_hicache_file_local import storage
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode


@pytest.mark.parametrize("metadata", [False, True])
def test_rank_and_checkpoint_isolation_after_reopen(tmp_path, metadata):
    expected = {}
    staging = torch.empty(1024, dtype=torch.int32)
    for rank in (0, 1):
        backend = storage(tmp_path, rank=rank, metadata=metadata)
        for checkpoint in range(8):
            for component in ("", ".mamba", ".qsa_indexer"):
                key = f"shared-prefix-branch-{checkpoint}{component}"
                salt = rank * 100000 + checkpoint * 1000 + len(component) * 10
                staging.copy_(torch.arange(1024, dtype=torch.int32) + salt)
                expected[rank, key] = staging.clone()
                assert backend.set(key, staging)
    for rank in (1, 0):
        backend = storage(tmp_path, rank=rank, metadata=metadata)
        for checkpoint in (7, 0, 5, 1, 6, 2, 4, 3, 0, 7):
            for component in (".qsa_indexer", "", ".mamba"):
                key = f"shared-prefix-branch-{checkpoint}{component}"
                staging.fill_(-999)
                result = backend.get(key, staging)
                assert result is not None
                assert torch.equal(result, expected[rank, key])
        staging.fill_(-999)
        assert backend.get("unknown-branch", staging) is None


def test_identical_key_replacement_does_not_change_other_rank(tmp_path):
    ranks = [storage(tmp_path, rank=rank) for rank in (0, 1)]
    for rank, backend in enumerate(ranks):
        assert backend.set("same-key", torch.full((64,), rank + 1, dtype=torch.int32))
    assert ranks[0].set("same-key", torch.full((64,), 17, dtype=torch.int32))
    # Existing keys may be immutable; either behavior must remain rank-local.
    result = storage(tmp_path, rank=1).get("same-key", torch.empty(64, dtype=torch.int32))
    assert torch.equal(result, torch.full((64,), 2, dtype=torch.int32))


@pytest.mark.parametrize("blocker", ["write", "load", "lock", "host_only", "device_only", "root"])
def test_writeback_cannot_reclaim_live_transfer_or_sole_copy(blocker):
    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core.root_node = UnifiedTreeNode((ComponentType.FULL,))
    node = UnifiedTreeNode((ComponentType.FULL,))
    node.parent = core.root_node
    data = node.component_data[ComponentType.FULL]
    data.value, data.host_value = torch.arange(64), torch.arange(64)
    assert core._can_reclaim_full_host_duplicate(node)
    if blocker == "write":
        node.write_through_pending_id = 17
    elif blocker == "load":
        node.load_back_pending_id = 17
    elif blocker == "lock":
        data.host_lock_ref = 1
    elif blocker == "host_only":
        data.value = None
    elif blocker == "device_only":
        data.host_value = None
    else:
        core.root_node = node
    assert not core._can_reclaim_full_host_duplicate(node)
    # Completion/unlock restores eligibility only when both copies still exist.
    if blocker in ("write", "load", "lock"):
        node.write_through_pending_id = node.load_back_pending_id = None
        data.host_lock_ref = 0
        assert core._can_reclaim_full_host_duplicate(node)

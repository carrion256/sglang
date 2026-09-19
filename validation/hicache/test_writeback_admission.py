"""Execute the installed scheduler admission method with synthetic cache fixtures."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import sglang
from sglang.srt.mem_cache.cache_diagnostics import CacheDiagnostics

from test_prefetch_namespace import cache_fixture


def scheduler_method():
    source = Path(sglang.__file__).parent / "srt/managers/scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prefetch_kvcache")
    scope = {}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[method.name]


def fixture(policy="write_back", backed=False, root=False, prefix_keys=False):
    cache = NS(
        is_backuped=Mock(return_value=backed), is_root=Mock(return_value=root),
        hicache_storage_pass_prefix_keys=prefix_keys,
        get_prefix_hash_values=Mock(return_value=["synthetic-prefix"]),
        get_last_hash_value=Mock(return_value="synthetic-anchor"),
        prefetch_from_storage=Mock(), cache_diagnostics=CacheDiagnostics(),
    )
    if policy != "legacy":
        cache.is_write_back = policy == "write_back"
    req = NS(rid="synthetic", last_host_node=7, extra_key="adapter", cache_salt="salt",
             prefix_indices=list(range(64)), host_hit_length=64,
             full_untruncated_fill_ids=list(range(512)),
             init_next_round_input=Mock(), _compute_max_prefix_len=Mock(return_value=448))
    return NS(enable_hicache_storage=True, tree_cache=cache), req


def test_device_only_writeback_anchor_attempts_storage_lookup():
    scheduler, req = fixture()
    scheduler_method()(scheduler, req)
    scheduler.tree_cache.prefetch_from_storage.assert_called_once()
    assert req.rid not in scheduler.tree_cache.cache_diagnostics.pending


@pytest.mark.parametrize("policy", ["write_back", "write_through", "write_through_selective", "legacy"])
@pytest.mark.parametrize("backed,root", [(False, False), (True, False), (False, True), (True, True)])
def test_policy_and_legacy_compatibility(policy, backed, root):
    scheduler, req = fixture(policy, backed, root)
    scheduler_method()(scheduler, req)
    expected = backed or root or policy == "write_back"
    assert scheduler.tree_cache.prefetch_from_storage.call_count == int(expected)
    assert (req.rid in scheduler.tree_cache.cache_diagnostics.pending) == (not expected)


@pytest.mark.parametrize("prefix_keys", [False, True])
@pytest.mark.parametrize("boundary", [128, 192, 448, 511])
def test_exact_lookup_suffix_logprob_boundary_and_namespace(prefix_keys, boundary):
    scheduler, req = fixture(prefix_keys=prefix_keys)
    req._compute_max_prefix_len.return_value = boundary
    scheduler_method()(scheduler, req)
    req.init_next_round_input.assert_called_once_with(scheduler.tree_cache, cow_mamba=False)
    req._compute_max_prefix_len.assert_called_once_with(512)
    scheduler.tree_cache.prefetch_from_storage.assert_called_once_with(
        req.rid, 7, list(range(128, boundary)), "synthetic-anchor",
        ["synthetic-prefix"] if prefix_keys else None,
        request_namespace=("adapter", "salt"),
    )


def test_disabled_storage_does_not_initialize_or_lookup():
    scheduler, req = fixture()
    scheduler.enable_hicache_storage = False
    scheduler_method()(scheduler, req)
    req.init_next_round_input.assert_not_called()
    scheduler.tree_cache.prefetch_from_storage.assert_not_called()


@pytest.mark.parametrize("blocker", ["none", "disabled", "no_controller", "short", "capacity", "namespace"])
def test_real_prefetch_keeps_its_rejection_and_reservation_contract(blocker):
    cache, node, _ = cache_fixture("unified", ("adapter", "salt"), root=False)
    scheduler, req = fixture()
    original = scheduler.tree_cache
    for name in ("is_backuped", "is_root", "hicache_storage_pass_prefix_keys",
                 "get_prefix_hash_values", "get_last_hash_value", "cache_diagnostics"):
        setattr(cache, name, getattr(original, name))
    cache.is_write_back = True
    controller = cache.cache_controller
    if blocker == "disabled":
        cache.tree_core.enable_storage = False
    elif blocker == "no_controller":
        cache.cache_controller = None
    elif blocker == "short":
        cache.prefetch_threshold = 1024
    elif blocker == "capacity":
        controller.prefetch_rate_limited = lambda: True
    elif blocker == "namespace":
        req.cache_salt = "other-salt"
    req.last_host_node = node
    scheduler.tree_cache = cache
    scheduler_method()(scheduler, req)
    if blocker != "none":
        assert not cache.ongoing_prefetch
        assert controller.prefetch_tokens_occupied == 0
        assert controller.prefetch_queue.empty()
        cache.inc_host_lock_ref.assert_not_called()
    else:
        pending = cache.ongoing_prefetch[req.rid]
        key = pending[1]
        assert list(key) == list(range(128, 448))
        assert (key.extra_key, key.cache_salt) == ("adapter", "salt")
        assert controller.prefetch_tokens_occupied == 320
        cache.inc_host_lock_ref.assert_called_once_with(node)
        assert not controller.prefetch_queue.empty()


@pytest.mark.parametrize("write_back", [False, True])
def test_device_only_anchor_host_pin_lifetime(write_back):
    from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
    node = UnifiedTreeNode((ComponentType.FULL,))
    component = NS(component_type=ComponentType.FULL,
                   tree_core=NS(is_write_back=write_back, _update_evictable_leaf_sets=Mock()))
    data = node.component_data[ComponentType.FULL]
    token = object()
    assert data.host_value is None
    for _ in range(2):
        assert FullComponent.acquire_component_lock(component, node, token, lock_host=True) is token
    assert data.host_lock_ref == (2 if write_back else 0)
    for _ in range(3):
        FullComponent.release_component_lock(component, node, None, lock_host=True)
    assert data.host_lock_ref == 0

"""Publication history describes past acknowledgements, never current disk residency."""
import logging
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.mem_cache.cache_diagnostics import CacheDiagnostics

from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
from sglang.srt.mem_cache.unified_cache.components import ComponentType

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


@pytest.mark.parametrize('prior,generation', [(0, 1), (0, 3), (1, 2), (2, 4)])
@pytest.mark.parametrize('reason,device,host,required', [
    ('missing_mamba', True, False, True),
    ('missing_mamba', False, False, True),
    ('missing_mamba', True, False, False),
    ('missing_hash', True, True, True),
    ('missing_hash', False, False, False),
])
def test_rejection_classifies_history_without_changing_accounting(
    caplog, prior, generation, reason, device, host, required
):
    caplog.set_level(logging.DEBUG)
    node = UnifiedTreeNode((FULL, MAMBA))
    node.hash_value = ['synthetic'] if reason == 'missing_mamba' else None
    node.component_data[FULL].host_value = torch.tensor([1])
    node.component_data[MAMBA].value = torch.tensor([2]) if device else None
    node.component_data[MAMBA].host_value = torch.tensor([3]) if host else None
    node.mamba_checkpoint_required = required
    node.mamba_storage_generation = generation
    node.mamba_storage_success_generation = prior
    node.mamba_storage_failures = {generation - 1}
    before = (generation, prior, set(node.mamba_storage_failures))
    diagnostics = CacheDiagnostics()
    core = NS(node_by_id=lambda _: node, components_by_type={MAMBA: NS(cache=NS(cache_diagnostics=diagnostics))})
    assert UnifiedTreeCore.build_storage_backup_spec(core, node.id, False) is None
    assert (node.mamba_storage_generation, node.mamba_storage_success_generation,
            node.mamba_storage_failures) == before
    records = [r for r in caplog.records if 'phase=publication' in r.getMessage()]
    quiet = reason == "missing_mamba" and prior > 0
    assert len(records) == 1 and records[0].levelno == (logging.DEBUG if quiet else logging.ERROR)
    assert diagnostics.repeat_publications == int(quiet)
    text = records[0].getMessage()
    history = 'prior_success' if prior else 'no_prior_success'
    for field in (f'reason={reason}', f'publication_history={history}',
                  f'generation={generation}', f'prior_success_generation={prior}',
                  f'device_mamba={device}', f'host_mamba={host}',
                  f'checkpoint_required={required}', 'storage_residency=unverified'):
        assert field in text



def test_counter_metrics(monkeypatch):
    import prometheus_client
    registry = prometheus_client.CollectorRegistry()
    original = prometheus_client.Counter
    monkeypatch.setattr(prometheus_client, 'Counter', lambda *a, **kw: original(*a, registry=registry, **kw))
    diagnostics = CacheDiagnostics(rank=1, dp_rank=2, metrics=True)
    diagnostics.record_repeat_publication()
    diagnostics.record_repeat_publication()
    assert diagnostics.repeat_publications == 2
    assert registry.get_sample_value('sglang:hicache_repeat_publication_rejections_total',
        {'tp_rank': '1', 'dp_rank': '2'}) == 2
    assert not diagnostics.pending and not diagnostics.counts


def test_counter_without_metrics():
    diagnostics = CacheDiagnostics(metrics=False)
    diagnostics.record_repeat_publication()
    diagnostics.record_repeat_publication()
    assert diagnostics.repeat_publications == 2
    assert diagnostics.repeat_publication_metric is None

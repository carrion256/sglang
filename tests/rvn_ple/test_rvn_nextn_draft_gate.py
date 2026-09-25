"""Serve-time gate: patch 0057 refuses NEXTN on an RVN text checkpoint.

``--speculative-algorithm NEXTN`` builds its draft layer from ``model.mtp.*``
tensors inside the target checkpoint. An RVN text checkpoint has none:
``normalize_text_config`` pins ``mtp_num_hidden_layers`` to 0,
``Qwen4ExpForCausalLM`` asserts ``rvn_mtp_count(config) == 0`` at construction
(models/qwen4_exp.py:3060), and the text loader sends every name through
``reject_non_text_weight_name``, which rejects ``model.mtp.*`` -- the shipped
candidate index holds 296,238 tensors and not one of them is MTP-named.

Left ungated the launch does not report that. The draft worker's architecture is
never remapped to ``Qwen4ExpForCausalLMMTP`` (that remap is keyed on the
multimodal arch, configs/model_config.py:747), so the draft worker rebuilds the
whole target model and -- because patch 0051 propagates ``ple_offload_embedding``
only for ``not is_draft_worker`` -- allocates the n-gram table on the device as a
47.68 GiB ``VocabParallelEmbedding`` and dies in CUDA out of memory during that
allocation.

Conventions (repo style):
- Code under test is the patched tree's single file, loaded with importlib;
  ``RVN_PLE_TREE`` points at the root the series was applied to with -p1, and the
  module skips when unset so a bare ``pytest tests/`` stays green unpatched.
- The predicate is pure, so refusals and negative cases need no GPU and no
  checkpoint. The one test that drives the real handler stubs the two
  arg-resolution helpers the handler imports for backend choices, which have
  nothing to do with this decision. On the preimage that test fails because the
  handler sails past the missing gate instead of refusing.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCH_0057 = REPO / "patches" / "0057-rvn-nextn-draft-gate.patch"
_HOOK_REL = Path("python/sglang/srt/arg_groups/speculative_hook.py")
_ADAPTER_REL = Path("python/sglang/srt/models/qwen4_exp_text_adapter.py")

RVN_ARCH = "Qwen4ExpForCausalLM"
RVN_MODEL_TYPE = "qwen4_exp_text"
MULTIMODAL_ARCH = "Qwen4ExpForConditionalGeneration"
# NEXTN resolves to EAGLE before the handler runs, in
# _resolve_speculative_algorithm_alias; that is the string the handler sees, and
# SpeculativeAlgorithm.handle_server_args routes EAGLE to _handle_eagle_family.
NEXTN = "EAGLE"


def _tree_root():
    tree = os.environ.get("RVN_PLE_TREE")
    if not tree:
        pytest.skip(
            "RVN_PLE_TREE unset: apply patches/0047..0057 to a tree root with "
            "-p1 to exercise the gate"
        )
    root = Path(tree)
    for rel in (_HOOK_REL, _ADAPTER_REL):
        if not (root / rel).is_file():
            raise AssertionError(f"RVN_PLE_TREE={tree} lacks {rel}")
    return root


TREE = _tree_root()


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, TREE / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hook = _load(_HOOK_REL, "rvn_nextn_gate_hook")
# The predicate imports its collaborators from sglang.srt.models by name, so
# hand it the tree's copy instead of whatever the interpreter has installed; that
# module is standard-library only by design.
tree_adapter = _load(_ADAPTER_REL, "rvn_nextn_gate_adapter")

_ADAPTER_NAME = "sglang.srt.models.qwen4_exp_text_adapter"


@pytest.fixture(autouse=True, scope="module")
def _adapter_from_tree():
    previous = sys.modules.get(_ADAPTER_NAME)
    sys.modules[_ADAPTER_NAME] = tree_adapter
    yield
    if previous is None:
        del sys.modules[_ADAPTER_NAME]
    else:
        sys.modules[_ADAPTER_NAME] = previous


def gate():
    """Patch 0057's predicate, fetched lazily so a tree without the patch fails
    as a missing gate rather than as a collection error."""
    predicate = getattr(hook, "rvn_text_bundled_draft_unsupported", None)
    assert callable(predicate), (
        "patch 0057 is not applied to this tree: speculative_hook.py has no "
        "rvn_text_bundled_draft_unsupported, so NEXTN is never refused before "
        "the draft worker allocates the n-gram table"
    )
    return predicate


def rvn_config(**overrides):
    """RVN text config shape: flat, text-only, zero MTP depth."""
    config = {
        "architectures": [RVN_ARCH],
        "model_type": RVN_MODEL_TYPE,
        "num_hidden_layers": 48,
        "mtp_num_hidden_layers": 0,
        "num_nextn_predict_layers": 0,
        "ple_layer_ids": [2],
    }
    config.update(overrides)
    return config


def eagle_server_args(**overrides):
    """Server args as _handle_eagle_family sees them for a NEXTN launch: all
    spec parameters explicit, so nothing past the gate has to be inferred."""
    server_args = SimpleNamespace(
        device="cuda",
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
        max_running_requests=48,
        speculative_algorithm=NEXTN,
        speculative_draft_model_path=None,
        speculative_num_steps=2,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=3,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        model_path="/models/rvn-qwen38-ple-nvfp4",
        revision="main",
    )
    server_args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(**rvn_config())
    )
    for name, value in overrides.items():
        setattr(server_args, name, value)
    return server_args


# ------------------------------------------------------------------- tests


def test_rvn_text_without_draft_weights_is_refused():
    reason = gate()(NEXTN, rvn_config(), None)
    assert reason is not None, "NEXTN on an MTP-free RVN checkpoint must be refused"
    assert "mtp_num_hidden_layers" in reason
    assert "model.mtp.*" in reason
    # The message has to carry the ways out, not just the complaint.
    assert "--speculative-draft-model-path" in reason
    assert "without speculative decoding" in reason


def test_refusal_is_raised_from_arg_resolution_not_from_the_worker():
    """The point of the gate is failing before any allocation exists: the
    handler -- what SpeculativeAlgorithm.handle_server_args calls for EAGLE --
    must raise, so no draft worker is ever constructed, and it must raise rather
    than quietly default the draft path to the target."""
    real_overrides = sys.modules.get("sglang.srt.arg_groups.overrides")
    sys.modules["sglang.srt.arg_groups.overrides"] = SimpleNamespace(
        attention_backends_of=lambda *_a, **_k: (None, None),
        resolved_view=lambda _sa: SimpleNamespace(
            disable_overlap_schedule=True,
            enable_dp_attention=False,
            page_size=1,
            attention_backend=None,
        ),
    )
    server_args = eagle_server_args()
    try:
        with pytest.raises(ValueError, match="model.mtp"):
            hook._handle_eagle_family(server_args)
    finally:
        if real_overrides is None:
            del sys.modules["sglang.srt.arg_groups.overrides"]
        else:
            sys.modules["sglang.srt.arg_groups.overrides"] = real_overrides
    assert server_args.speculative_draft_model_path is None


@pytest.mark.parametrize(
    "algorithm,config,draft_path,why",
    [
        pytest.param(
            NEXTN,
            rvn_config(num_nextn_predict_layers=1),
            None,
            "the checkpoint declares the draft layer",
            id="rvn-declares-nextn-layers",
        ),
        pytest.param(
            NEXTN,
            rvn_config(mtp_num_hidden_layers=1),
            None,
            "the checkpoint declares the draft layer",
            id="rvn-declares-mtp-hidden-layers",
        ),
        pytest.param(
            NEXTN,
            rvn_config(),
            "/models/some-draft",
            "an external draft brings its own weights",
            id="explicit-draft-path",
        ),
        pytest.param(
            "STANDALONE",
            rvn_config(),
            None,
            "a standalone draft is trained, not bundled",
            id="standalone-algorithm",
        ),
        pytest.param(
            NEXTN,
            rvn_config(architectures=[MULTIMODAL_ARCH], model_type="qwen4_exp"),
            None,
            "the LIL multimodal path must keep working untouched",
            id="lil-multimodal-untouched",
        ),
        pytest.param(
            NEXTN,
            {
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "num_nextn_predict_layers": 1,
            },
            None,
            "no other architecture is named by the gate",
            id="other-arch-untouched",
        ),
        pytest.param(
            None, rvn_config(), None, "speculative decoding is off", id="no-spec"
        ),
    ],
)
def test_gate_leaves_every_other_launch_alone(algorithm, config, draft_path, why):
    assert gate()(algorithm, config, draft_path) is None, (
        f"gate must stay out of the way: {why}"
    )


def test_gate_reuses_the_text_adapters_own_mtp_definition():
    """The gate must not invent its own notion of MTP depth: RVN defines
    MTP-free as mtp_num_hidden_layers + num_nextn_predict_layers == 0, which is
    what the entry class asserts, so anything the gate lets through is a
    checkpoint the draft loader could actually fill."""
    assert tree_adapter.rvn_mtp_count(rvn_config()) == 0
    assert tree_adapter.rvn_mtp_count(rvn_config(num_nextn_predict_layers=1)) == 1
    assert tree_adapter.rvn_mtp_count(rvn_config(mtp_num_hidden_layers=2)) == 2
    assert (
        gate()(
            NEXTN, rvn_config(mtp_num_hidden_layers=2, num_nextn_predict_layers=1), None
        )
        is None
    )


def test_gate_lives_in_the_patched_tree():
    """Guard the red tail: an unpatched tree has no predicate, so a stacked tree
    that lost patch 0057 must not read as green."""
    assert "def rvn_text_bundled_draft_unsupported(" in (TREE / _HOOK_REL).read_text()
    assert PATCH_0057.is_file(), f"missing {PATCH_0057}"

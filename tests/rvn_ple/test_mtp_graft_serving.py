"""Serve-time support for a grafted MTP draft head: patches 0058 + 0059.

The base RVN candidate (``/models/rvn-qwen38-ple-nvfp4``) ships no draft layer,
and the text contract says so three times over: ``assert_rvn_text_config``
refuses a non-zero MTP count, ``Qwen4ExpForCausalLM.__init__`` then asserts
``rvn_mtp_count(config) == 0``, and the loader runs every weight name through
``reject_non_text_weight_name``, which raises on any name containing ``mtp``.
A *grafted* copy of that candidate does ship one -- base shards hardlinked, the
source checkpoint's top-level ``mtp.*`` tensors copied byte-identically, and a
``config.json`` that declares ``mtp_num_hidden_layers: 1`` plus the graft tool's
stamp ``rvn_mtp_graft = {source, encoder_version, count}`` -- and NEXTN has to
serve it. NEXTN is lossless against the target, so a weakly-matched draft can
cost acceptance rate but never correctness; that makes this a loader/selection
question, not a training one.

What is pinned here, in order of blast radius:

  * the one new authority, ``rvn_mtp_graft_active`` -- count == 1 AND the
    stamp's encoder_version == "rvn-mtp-graft-r1" AND its count == 1;
  * every other shape keeps the old raise verbatim (count > 1, unstamped,
    foreign encoder_version, stamp count disagreeing with the declared count);
  * the stamp relaxes the *raise*, never the *skip*: the target loader still
    drops every mtp-named tensor, so the text entry class never allocates a
    draft head;
  * the strict default of ``reject_non_text_weight_name`` (and therefore
    ``claim_text_weight_name``, the inventory tooling and the other suites)
    is unchanged;
  * the draft-selection half: ``configs/model_config.py`` remaps a grafted RVN
    text draft to ``Qwen4ExpForCausalLMMTP``, and ``arg_groups/
    speculative_hook.py`` defaults the draft path to the target path -- above
    patch 0057's gate, which keeps refusing an *unstamped* RVN text launch.

Conventions (repo style):
- Code under test is the patched tree's own files, loaded with
  ``importlib``/``ast``/``exec`` extraction; never ``import sglang`` host-scope.
- ``RVN_PLE_TREE`` is the root of the tree patches 0047..0059 were applied to
  with ``-p1``; the module skips when unset so ``pytest tests/`` stays green on
  unpatched trees.
- Configs are synthetic dicts/SimpleNamespaces; no GPU, no torch, no /models.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCH_0058 = REPO / "patches" / "0058-rvn-mtp-graft-loader.patch"
PATCH_0059 = REPO / "patches" / "0059-rvn-mtp-draft-remap.patch"

_ADAPTER_REL = Path("python/sglang/srt/models/qwen4_exp_text_adapter.py")
_MODEL_REL = Path("python/sglang/srt/models/qwen4_exp.py")
_CONFIG_REL = Path("python/sglang/srt/configs/model_config.py")
_HOOK_REL = Path("python/sglang/srt/arg_groups/speculative_hook.py")

RVN_ARCH = "Qwen4ExpForCausalLM"
RVN_MODEL_TYPE = "qwen4_exp_text"
MULTIMODAL_ARCH = "Qwen4ExpForConditionalGeneration"
MTP_DRAFT_ARCH = "Qwen4ExpForCausalLMMTP"
# The frozen graft encoder version, from the graft contract. Deliberately a
# second literal here: adapter.RVN_MTP_GRAFT_ENCODER_VERSION is compared to it.
GRAFT_ENCODER_VERSION = "rvn-mtp-graft-r1"
# The handler only ever sees the resolved algorithm name: NEXTN is rewritten to
# EAGLE before SpeculativeAlgorithm.handle_server_args dispatches.
EAGLE = "EAGLE"
STANDALONE = "STANDALONE"

# Real tensor names from the graft contract (the source checkpoint's mtp.* set),
# covering the three shapes the draft's load_weights rewrites: top-level fusion
# parameters, pre-fc norms, and the single draft decoder layer.
GRAFTED_NAMES = (
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.hyper_connection_mixer.down_proj.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.experts.3.gate_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
)


def _tree_root():
    tree = os.environ.get("RVN_PLE_TREE")
    if not tree:
        pytest.skip(
            "RVN_PLE_TREE unset: apply patches/0047..0059 to a tree root with "
            "-p1 to exercise the grafted-draft contract",
            allow_module_level=True,
        )
    root = Path(tree)
    for rel in (_ADAPTER_REL, _MODEL_REL, _CONFIG_REL, _HOOK_REL):
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


adapter = _load(_ADAPTER_REL, "rvn_mtp_graft_adapter")
hook = _load(_HOOK_REL, "rvn_mtp_graft_hook")

_ADAPTER_NAME = "sglang.srt.models.qwen4_exp_text_adapter"


@pytest.fixture(autouse=True, scope="module")
def _adapter_from_tree():
    """The predicate, the loader and the draft remap all import their
    collaborators from ``sglang.srt.models`` by name; hand them this tree's
    copy instead of whatever the interpreter has installed. The adapter module
    is standard-library only by design."""
    previous = sys.modules.get(_ADAPTER_NAME)
    sys.modules[_ADAPTER_NAME] = adapter
    yield
    if previous is None:
        del sys.modules[_ADAPTER_NAME]
    else:
        sys.modules[_ADAPTER_NAME] = previous


# ------------------------------------------------------------- config fixtures


def graft_stamp(**overrides):
    stamp = {
        "source": "/models/qwen38-flash-next",
        "encoder_version": GRAFT_ENCODER_VERSION,
        "count": 1,
    }
    stamp.update(overrides)
    return stamp


def rvn_config(**overrides):
    """The shipped candidate's config shape: flat, text-only, zero MTP depth,
    plus the inert ``mtp`` sub-object the producer emits (``rvn_mtp_count``
    deliberately ignores it)."""
    config = {
        "architectures": [RVN_ARCH],
        "model_type": RVN_MODEL_TYPE,
        "num_hidden_layers": 48,
        "hidden_size": 2560,
        "mtp_num_hidden_layers": 0,
        "ple_layer_ids": [2],
        "hc_count": 4,
        "mtp": {"hybrid": True, "num_hidden_layers": 0},
    }
    config.update(overrides)
    return config


def grafted_config(**overrides):
    config = rvn_config(mtp_num_hidden_layers=1, rvn_mtp_graft=graft_stamp())
    config.update(overrides)
    return config


def graft_active():
    predicate = getattr(adapter, "rvn_mtp_graft_active", None)
    assert callable(predicate), (
        "patch 0058 is not applied to this tree: qwen4_exp_text_adapter.py has "
        "no rvn_mtp_graft_active, so a grafted mtp.* head cannot be served"
    )
    return predicate


def zero_count_message():
    """The one message every refused shape must still carry, byte for byte."""
    return "RVN text config must declare zero MTP layers"


# ------------------------------------------------------- graft-active predicate


def test_graft_encoder_version_literal_is_shared():
    """The adapter's frozen literal and this test's must not drift: a renamed
    encoder would silently un-stamp every grafted checkpoint."""
    assert adapter.RVN_MTP_GRAFT_ENCODER_VERSION == GRAFT_ENCODER_VERSION


def test_zero_count_is_never_a_graft():
    """The base candidate stays an ordinary text checkpoint: the stamp alone is
    not enough, the declared count has to be 1."""
    assert graft_active()(rvn_config()) is False
    assert graft_active()(rvn_config(rvn_mtp_graft=graft_stamp())) is False


def test_unstamped_single_layer_is_not_a_graft():
    """An operator who edits mtp_num_hidden_layers by hand without running the
    graft tool gets the old refusal, not a draft head that does not exist."""
    assert graft_active()(rvn_config(mtp_num_hidden_layers=1)) is False


def test_stamp_with_wrong_count_is_not_a_graft():
    """The stamp's own count must agree with the declared depth, so a graft of
    two layers cannot be served as if it had one."""
    assert graft_active()(grafted_config(rvn_mtp_graft=graft_stamp(count=2))) is False
    assert graft_active()(grafted_config(rvn_mtp_graft=graft_stamp(count=0))) is False
    assert graft_active()(grafted_config(num_nextn_predict_layers=1)) is False
    assert graft_active()(rvn_config(mtp_num_hidden_layers=2,
                                     rvn_mtp_graft=graft_stamp())) is False


def test_foreign_encoder_version_is_not_a_graft():
    for version in ("rvn-mtp-graft-r2", "rvn-ple-nvfp4-r1", "", None):
        assert graft_active()(
            grafted_config(rvn_mtp_graft=graft_stamp(encoder_version=version))
        ) is False, version


def test_stamped_single_layer_is_a_graft():
    assert graft_active()(grafted_config()) is True
    # The stamp is a mapping of exactly the three contract fields; extra keys
    # (a future graft carrying a digest, say) must not unstamp it.
    assert graft_active()(
        grafted_config(rvn_mtp_graft={**graft_stamp(), "shard_count": 3})
    ) is True
    # Attribute-style configs (a PretrainedConfig, not a dict) read the same way.
    assert graft_active()(SimpleNamespace(**grafted_config())) is True


# ---------------------------------------------------- config-level acceptance


def test_unstamped_single_layer_is_still_refused():
    """Red case: the candidate that ships no draft layer must keep raising the
    message it raised before patch 0058 existed."""
    with pytest.raises(ValueError) as exc:
        adapter.assert_rvn_text_config(rvn_config(mtp_num_hidden_layers=1))
    assert str(exc.value).startswith(zero_count_message())


def test_stamped_single_layer_is_accepted():
    adapter.assert_rvn_text_config(grafted_config())  # must not raise


@pytest.mark.parametrize(
    "config,why",
    [
        pytest.param(rvn_config(mtp_num_hidden_layers=2), "count 2, stamped", id="two"),
        pytest.param(
            grafted_config(num_nextn_predict_layers=1),
            "declared 1 + nextn 1 = 2",
            id="sums-to-two",
        ),
        pytest.param(
            grafted_config(rvn_mtp_graft=graft_stamp(encoder_version="rvn-mtp-graft-r9")),
            "wrong encoder_version",
            id="foreign-encoder",
        ),
        pytest.param(
            grafted_config(rvn_mtp_graft=graft_stamp(count=2)),
            "stamp count disagrees",
            id="stamp-count-two",
        ),
        pytest.param(
            grafted_config(rvn_mtp_graft=graft_stamp(count=3)),
            "stamp claims three layers",
            id="stamp-count-three",
        ),
        pytest.param(
            grafted_config(rvn_mtp_graft=None), "stamp removed again", id="no-stamp",
        ),
        pytest.param(
            grafted_config(rvn_mtp_graft={"source": "/models/qwen38-flash-next"}),
            "stamp without an encoder_version",
            id="stamp-without-version",
        ),
    ],
)
def test_every_other_non_zero_shape_keeps_the_refusal(config, why):
    """Patch 0058 opens exactly one hole, and it is a square one: any other
    non-zero MTP declaration keeps the pre-0058 message verbatim, so an
    operator who half-does the graft gets the same error they got before."""
    with pytest.raises(ValueError) as exc:
        adapter.assert_rvn_text_config(config)
    assert str(exc.value).startswith(zero_count_message()), why


def test_assert_still_refuses_non_rvn_and_vision_configs():
    """The graft work must not have loosened the neighbours."""
    with pytest.raises(ValueError, match="not an RVN text config"):
        adapter.assert_rvn_text_config(
            rvn_config(architectures=[MULTIMODAL_ARCH])
        )
    # A stamp cannot rescue a config that is not RVN text at all: detection is
    # architecture *and* model_type, and neither is loosened for the graft.
    with pytest.raises(ValueError, match="not an RVN text config"):
        adapter.assert_rvn_text_config(grafted_config(model_type="qwen4_exp"))
    with pytest.raises(ValueError, match="must not enable vision fields"):
        adapter.assert_rvn_text_config(
            grafted_config(vision_config={"depth": 32})
        )


# --------------------------------------------- normalization (the 0-pin rule)


def test_normalization_pins_the_base_candidate_to_zero():
    """Unstamped behaviour is untouched: count 0 in, count 0 out, and every
    preserved field keeps its incoming value."""
    out = adapter.normalize_text_config(rvn_config())
    assert out["mtp_num_hidden_layers"] == 0
    assert out["ple_layer_ids"] == [2]
    assert out["hc_count"] == 4
    assert out["num_hidden_layers"] == 48


def test_normalization_keeps_the_grafted_layer():
    """The 0-pin is what would hide the grafted layer from the NEXTN remap in
    configs/model_config.py, so a stamped config must come back with 1."""
    out = adapter.normalize_text_config(grafted_config())
    assert out["mtp_num_hidden_layers"] == 1
    assert out["rvn_mtp_graft"]["encoder_version"] == GRAFT_ENCODER_VERSION
    assert out["ple_layer_ids"] == [2]  # preserved fields still preserved


def test_normalization_refuses_the_shapes_assert_refuses():
    for config in (
        rvn_config(mtp_num_hidden_layers=1),
        grafted_config(rvn_mtp_graft=graft_stamp(encoder_version="rvn-ple-nvfp4-r1")),
    ):
        with pytest.raises(ValueError) as exc:
            adapter.normalize_text_config(config)
        assert str(exc.value).startswith(zero_count_message())


# ------------------------------------------------- loader name rule (skip/raise)


@pytest.mark.parametrize("name", GRAFTED_NAMES)
def test_grafted_names_pass_the_gate_and_are_skipped(name):
    """The target loader's contract for a grafted checkpoint: the raise goes
    away, the skip does not. ``allow_mtp=True`` is what
    ``rvn_mtp_graft_active(self.config)`` evaluates to in
    ``Qwen4ExpWeightLoadMixin.load_qwen4_exp_weights``, and the very next
    statement in that loop is ``if "mtp" in name: continue``, so these tensors
    are dropped rather than allocated -- which is also what
    ``test_loader_keeps_the_mtp_skip`` pins structurally."""
    adapter.reject_non_text_weight_name(name, allow_mtp=True)  # must not raise
    with pytest.raises(ValueError, match="must not contain MTP weights"):
        adapter.reject_non_text_weight_name(name)  # the strict default


def test_visual_names_are_refused_even_under_the_stamp():
    """The escape hatch is for mtp.* only; a visual tensor in a text checkpoint
    is still a corrupt graft."""
    for name in (
        "model.language_model.visual.patch_embed.proj.weight",
        "visual.blocks.0.attn.qkv.weight",
    ):
        with pytest.raises(ValueError, match="must not contain visual weights"):
            adapter.reject_non_text_weight_name(name, allow_mtp=True)


def test_text_names_are_unaffected_by_the_new_keyword():
    adapter.reject_non_text_weight_name("model.language_model.layers.7.mlp.gate_proj.weight")
    adapter.reject_non_text_weight_name(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight",
        allow_mtp=True,
    )


def test_claim_rule_stays_strict():
    """claim_text_weight_name (the inventory/loader claim rule, and the one the
    other suites exercise) never passes allow_mtp, so an mtp name still raises
    there: the grafted tensors are the draft worker's, never the target's."""
    with pytest.raises(ValueError, match="must not contain MTP weights"):
        adapter.claim_text_weight_name("mtp.fc.weight")
    assert adapter.claim_text_weight_name(
        "model.language_model.layers.1.ple.ple_embedding"
        ".ngram_embedding.shard_3.weight"
    ) == "ple_shard"


# ------------------------------------------------------- loader wiring (AST)


def _function(rel, class_name, func_name):
    module = ast.parse((TREE / rel).read_text())
    scope = module
    if class_name is not None:
        scope = next(
            n
            for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    return next(
        n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == func_name
    )


def test_loader_wires_allow_mtp_to_the_stamp():
    loader = _function(_MODEL_REL, "Qwen4ExpWeightLoadMixin", "load_qwen4_exp_weights")
    calls = [
        n
        for n in ast.walk(loader)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "reject_non_text_weight_name"
    ]
    assert len(calls) == 1, "the text-only gate must stay a single call site"
    (keyword,) = [kw for kw in calls[0].keywords if kw.arg == "allow_mtp"]
    # allow_mtp is fed by a pre-loop local (the stamp is config-derived, so it
    # is read once rather than per tensor). Resolve that name and check what it
    # actually holds: the stamp predicate AND text_only -- never a bare True,
    # and never the declared count alone, which is only half of the rule.
    fed_by = keyword.value
    assert isinstance(fed_by, ast.Name), ast.unparse(fed_by)
    assignments = [
        n
        for n in ast.walk(loader)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == fed_by.id for t in n.targets)
    ]
    assert len(assignments) == 1, fed_by.id
    value = assignments[0].value
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "rvn_mtp_graft_active"
        for n in ast.walk(value)
    ), "allow_mtp must be keyed on rvn_mtp_graft_active(config)"
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "config" for n in ast.walk(value)
    ), "the predicate must read this model's own config"
    assert any(
        isinstance(n, ast.Name) and n.id == "text_only" for n in ast.walk(value)
    ), "the multimodal path must never get the escape hatch"


def test_loader_keeps_the_mtp_skip():
    """The stamp relaxes the raise, not the skip: if ``if "mtp" in name:
    continue`` ever goes away, a grafted checkpoint would try to build the draft
    head's parameters inside the target model."""
    loader = _function(_MODEL_REL, "Qwen4ExpWeightLoadMixin", "load_qwen4_exp_weights")
    skips = [
        n
        for n in ast.walk(loader)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Constant)
        and n.test.left.value == "mtp"
        and any(isinstance(s, ast.Continue) for s in n.body)
    ]
    assert skips, "mtp-named tensors must still be skipped by the target loader"


def test_entry_class_assert_tolerates_only_the_stamp():
    init = _function(_MODEL_REL, "Qwen4ExpForCausalLM", "__init__")
    asserts = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assert)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "rvn_mtp_count"
            for c in ast.walk(n.test)
        )
    ]
    assert len(asserts) == 1
    (condition,) = asserts
    assert any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "rvn_mtp_graft_active"
        for c in ast.walk(condition.test)
    ), "the entry class must allow count 1 only under the graft stamp"


# --------------------------------------------------- draft remap (model_config)


def _module_function_source(rel, name):
    source = (TREE / rel).read_text()
    module = ast.parse(source)
    node = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def _draft_remap_branch():
    """The ModelConfig branch that remaps a grafted RVN text draft."""
    module = ast.parse((TREE / _CONFIG_REL).read_text())
    config_class = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ModelConfig"
    )
    for node in ast.walk(config_class):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "is_draft_model" in test and "is_rvn_text_mtp_graft" in test:
            return node
    raise AssertionError(
        "patch 0059 is not applied to this tree: ModelConfig has no draft branch "
        "keyed on is_rvn_text_mtp_graft, so the draft worker is built as a second "
        "copy of the 48-layer target"
    )


def test_graft_helper_keys_on_arch_and_stamp():
    """Behavioural, not textual: the helper's own source is exec'd with the
    tree adapter behind the lazy import, so the architecture guard and the
    stamp guard are both really consulted."""
    namespace = {
        "_hf_arch": lambda config: (
            (config.get("architectures") if isinstance(config, dict)
             else getattr(config, "architectures", None)) or [None]
        )[0]
    }
    exec(compile(_module_function_source(_CONFIG_REL, "is_rvn_text_mtp_graft"),
                 str(TREE / _CONFIG_REL), "exec"), namespace)
    helper = namespace["is_rvn_text_mtp_graft"]
    assert helper(grafted_config()) is True
    assert helper(rvn_config()) is False
    assert helper(rvn_config(mtp_num_hidden_layers=1)) is False
    # A stamp on some other architecture must not remap it: the graft contract
    # is an RVN-text-only contract.
    assert helper(rvn_config(architectures=[MULTIMODAL_ARCH],
                             mtp_num_hidden_layers=1,
                             rvn_mtp_graft=graft_stamp())) is False
    assert helper({}) is False


def test_draft_remap_mirrors_the_multimodal_mtp_branch():
    """Everything the draft worker needs is set here, and only here: the MTP
    architecture, a deep copy so the target keeps its depth, and the one-layer
    full_attention collapse that sizes the draft KV pool
    (model_executor/model_runner_components/layer_setup.py reads
    num_nextn_predict_layers)."""
    branch = _draft_remap_branch()
    written = {}
    for node in ast.walk(branch):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            key = ast.unparse(target)
            written.setdefault(key, set()).add(ast.unparse(node.value))
    assert "self.hf_config.architectures[0]" in written
    assert {f"'{MTP_DRAFT_ARCH}'"} == written["self.hf_config.architectures[0]"]
    assert "self.hf_config" in written and any(
        v.startswith("copy.deepcopy(") for v in written["self.hf_config"]
    ), "the target shares this hf_config object; the draft must copy it first"
    for field, value in (
        ("num_nextn_predict_layers", "1"),
        ("num_hidden_layers", "1"),
        ("full_attention_interval", "1"),
    ):
        key = f"text_config.{field}"
        assert key in written and written[key] == {value}, key
    assert written.get("text_config.layer_types") == {"['full_attention']"}
    assert "self.hf_text_config = get_hf_text_config(self.hf_config)" in ast.unparse(
        branch
    )


# ---------------------------------------------------- draft path (serve hook)


def _overrides_stub():
    return SimpleNamespace(
        attention_backends_of=lambda *_a, **_k: (None, None),
        resolved_view=lambda _sa: SimpleNamespace(
            disable_overlap_schedule=True,
            enable_dp_attention=False,
            page_size=1,
            attention_backend="flashinfer",
        ),
    )


def eagle_server_args(config, **overrides):
    """Server args as _handle_eagle_family sees them for a NEXTN launch: every
    spec parameter explicit, so nothing past the draft-path default has to be
    inferred and the handler walks to its end."""
    server_args = SimpleNamespace(
        device="cuda",
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
        max_running_requests=48,
        speculative_algorithm=EAGLE,
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        model_path="/models/rvn-qwen38-ple-nvfp4-mtp",
        revision="main",
    )
    server_args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(**config)
    )
    for name, value in overrides.items():
        setattr(server_args, name, value)
    return server_args


def _run_eagle_handler(server_args):
    """_handle_eagle_family with its only heavy collaborations stubbed, the way
    test_rvn_nextn_draft_gate.py does for the 0057 gate."""
    previous = sys.modules.get("sglang.srt.arg_groups.overrides")
    sys.modules["sglang.srt.arg_groups.overrides"] = _overrides_stub()
    try:
        hook._handle_eagle_family(server_args)
    finally:
        if previous is None:
            del sys.modules["sglang.srt.arg_groups.overrides"]
        else:
            sys.modules["sglang.srt.arg_groups.overrides"] = previous


def test_grafted_launch_defaults_the_draft_path_to_the_target():
    """The draft layer lives in the grafted directory itself, so the launch
    needs no --speculative-draft-model-path: managers/tp_worker.py reads the
    draft worker's weights from exactly this path."""
    server_args = eagle_server_args(grafted_config())
    _run_eagle_handler(server_args)
    assert server_args.speculative_draft_model_path == server_args.model_path
    assert server_args.speculative_draft_model_revision == "main"


def test_ungrafted_launch_is_still_refused_before_any_default():
    """Patch 0059 must not have silenced patch 0057. The refusal is what keeps
    the candidate from allocating a second 47.68 GiB n-gram table in a draft
    worker that can never read it, and it still has to happen above the
    draft-path defaulting."""
    server_args = eagle_server_args(rvn_config())
    with pytest.raises(ValueError, match="model.mtp"):
        _run_eagle_handler(server_args)
    assert server_args.speculative_draft_model_path is None


def test_gate_predicate_itself_is_unchanged_for_a_graft():
    """Sharp version of the ordering: reached *with* no draft path, a grafted
    config is still refused by 0057's predicate -- so the launch only works
    because the default runs above the gate. If the default ever moves below
    it, this pair fails instead of resurrecting the OOM."""
    gate = getattr(hook, "rvn_text_bundled_draft_unsupported", None)
    assert callable(gate), "patch 0057's gate is missing from this tree"
    assert gate(EAGLE, grafted_config(), None) is not None
    assert gate(EAGLE, grafted_config(), "/models/elsewhere") is None


def test_standalone_is_not_given_the_bundled_draft_path():
    """The default is scoped to the resolved EAGLE name (what NEXTN arrives as)
    exactly like 0057's predicate. STANDALONE brings its own draft weights and
    must keep failing for a missing --speculative-draft-model-path instead of
    quietly pointing at the target."""
    server_args = eagle_server_args(grafted_config(),
                                    speculative_algorithm=STANDALONE)
    _run_eagle_handler(server_args)
    assert server_args.speculative_draft_model_path is None


def test_stamp_without_the_rvn_text_model_type_is_not_defaulted():
    """Both halves of the RVN text contract are required: architectures alone
    would let a stamped multimodal or renamed config borrow the default."""
    server_args = eagle_server_args(
        grafted_config(model_type="qwen4_exp", architectures=[RVN_ARCH])
    )
    _run_eagle_handler(server_args)
    assert server_args.speculative_draft_model_path is None


def test_explicit_draft_path_survives_the_graft():
    """An operator who points --speculative-draft-model-path somewhere else
    keeps it; the default must not overwrite a value that is already set."""
    server_args = eagle_server_args(
        grafted_config(), speculative_draft_model_path="/models/other-draft"
    )
    _run_eagle_handler(server_args)
    assert server_args.speculative_draft_model_path == "/models/other-draft"


# --------------------------------------------------------------- tree hygiene


def test_patches_exist_and_are_tracked_in_the_repo():
    """Red tail: an unpatched tree has none of the symbols above, so pin that
    the two patches this suite covers are the ones being registered."""
    assert PATCH_0058.is_file(), f"missing {PATCH_0058}"
    assert PATCH_0059.is_file(), f"missing {PATCH_0059}"


def test_draft_class_needs_no_ple_table():
    """The whole reason a 1-layer draft can share the target directory:
    Qwen4ExpForCausalLMMTP clears ple_layer_ids itself, so the grafted draft
    never asks for the 26.8 GiB n-gram table the target pins in host RAM."""
    mtp_rel = Path("python/sglang/srt/models/qwen4_exp_mtp.py")
    if not (TREE / mtp_rel).is_file():
        pytest.skip(f"{mtp_rel} is not in this tree")
    module = ast.parse((TREE / mtp_rel).read_text())
    draft = next(
        n for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == MTP_DRAFT_ARCH
    )
    init = next(n for n in draft.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    clears = [
        n for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t).endswith("ple_layer_ids") for t in n.targets)
        and ast.unparse(n.value) == "[]"
    ]
    assert clears, "the draft must clear ple_layer_ids, or it needs a PLE table"
    # And it inherits its weight-name mapping from Qwen3_5ForCausalLMMTP, which
    # is why the graft's top-level mtp.* names need no text-path remapping.
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "load_weights"
        for n in draft.body
    ), "the draft must keep the inherited mtp.* name mapping"

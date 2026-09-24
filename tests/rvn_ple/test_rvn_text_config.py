"""WP1: RVN text-config conformance for patches/0047-rvn-text-config.patch.

Applies the patch to a scratch mirror of the deployed preimage tree
(git apply --unsafe-paths) and exercises the dependency-light
``qwen4_exp_text_adapter`` helper via importlib (no sglang/torch imports), plus
AST checks on the patched model file.

Preimage roots (first existing wins): $RVN_PREIMAGE, /pre,
/tmp/rvn-preimage-sglang, /sgl-workspace/sglang/python/sglang.
"""

import ast
import importlib.util
import os
import py_compile
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "patches" / "0047-rvn-text-config.patch"

TOUCHED = (
    "python/sglang/srt/models/qwen4_exp.py",
    "python/sglang/srt/utils/hf_transformers/common.py",
)
HELPER = "python/sglang/srt/models/qwen4_exp_text_adapter.py"

_PREIMAGE_CANDIDATES = (
    os.environ.get("RVN_PREIMAGE"),
    "/pre",
    "/tmp/rvn-preimage-sglang",
    "/sgl-workspace/sglang/python/sglang",
)

# Verified-live RVN checkpoint config fields (0bserverx/RVN-...-NVFP4).
RVN_CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp_text",
    "tie_word_embeddings": False,
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "num_experts": 512,
    "full_attention_interval": 4,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "hc_count": 4,
    "hc_lowrank": 320,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 200000000,
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": 128,
    "seed": 1234,
    "ple_layer_ids": [2],
    "ple_embed_dim": 1024,
    "ple_conv_kernel_size": 4,
    "ple_offload_embedding": False,
    "ple_embedding_dtype": None,
    "index_share_for_mtp_iteration": False,
    "indexer_budget": 1.0,
    "indexer_compress_ratio": 8,
    "indexer_head_dim": 128,
    "indexer_kv_heads": 2,
    "indexer_n_heads": 4,
    "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
    "mtp_num_hidden_layers": 0,
    "mtp_use_dedicated_embeddings": False,
    "mtp": {"num_nextn_predict_layers": 0, "hidden_size": 4096},
}

# Equivalent LIL (multimodal) config: nested text_config + vision_config.
LIL_CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "tie_word_embeddings": False,
    "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "out_hidden_size": 4096,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [8, 16, 24],
    },
    "text_config": {
        "model_type": "qwen4_exp_text",
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "hc_count": 4,
        "hc_lowrank": 320,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 200000000,
        "split_ngram_parts": 128,
        "ple_layer_ids": [2],
        "indexer_budget": 1.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    },
    "mtp": {"num_nextn_predict_layers": 1, "hidden_size": 4096},
}


def _preimage_root():
    for cand in _PREIMAGE_CANDIDATES:
        if cand and (Path(cand) / "srt/models/qwen4_exp.py").is_file():
            return Path(cand)
    pytest.fail(
        "no Qwen4Exp preimage tree found; set RVN_PREIMAGE to the extracted "
        "deployed tree (must contain srt/models/qwen4_exp.py)"
    )


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Scratch tree: preimage originals + patch applied via git apply."""
    root = tmp_path_factory.mktemp("rvn0047-tree")
    pre = _preimage_root()
    for rel in TOUCHED:
        src = pre / rel.removeprefix("python/sglang/")
        assert src.is_file(), f"missing preimage file {src}"
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    assert PATCH.is_file(), f"missing patch {PATCH}"
    for cmd in (
        ["git", "apply", "--check", "--unsafe-paths", str(PATCH)],
        ["git", "apply", "--unsafe-paths", str(PATCH)],
    ):
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{' '.join(cmd)} failed:\n{proc.stderr}"
    return root


@pytest.fixture(scope="module")
def adapter(tree):
    path = tree / HELPER
    assert path.is_file(), "patch must add the qwen4_exp_text_adapter helper"
    spec = importlib.util.spec_from_file_location("rvn_text_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(tree, rel):
    return ast.parse((tree / rel).read_text())


def _entry_class_names(tree, rel):
    module = _parse(tree, rel)
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "EntryClass" for t in node.targets
            )
            and isinstance(node.value, ast.List)
        ):
            return [
                e.id if isinstance(e, ast.Name) else getattr(e, "value", None)
                for e in node.value.elts
            ]
    return []


def _class_body(tree, rel, name):
    module = _parse(tree, rel)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {rel}")


def _function(tree, rel, class_name, func_name):
    cls = _class_body(tree, rel, class_name)
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == func_name
        ):
            return node
    raise AssertionError(f"{class_name}.{func_name} not found in {rel}")


# ---------------------------------------------------------------------------
# Detection and config normalization
# ---------------------------------------------------------------------------


def test_rvn_config_resolves_text_only(adapter):
    assert adapter.is_rvn_text_model(RVN_CONFIG["architectures"], RVN_CONFIG["model_type"])
    normalized = adapter.normalize_text_config(RVN_CONFIG)
    # MTP count 0, no vision fields enabled.
    assert adapter.rvn_mtp_count(normalized) == 0
    assert normalized["mtp_num_hidden_layers"] == 0
    assert "vision_config" not in normalized
    assert "text_config" not in normalized  # flat stays flat
    assert normalized["mtp"] == RVN_CONFIG["mtp"]  # nested inert object preserved


def test_rvn_config_preserves_attention_and_ngram_fields(adapter):
    normalized = adapter.normalize_text_config(RVN_CONFIG)
    for field in adapter.PRESERVED_CONFIG_FIELDS:
        if field in RVN_CONFIG:
            assert normalized[field] == RVN_CONFIG[field], field
    # Named families explicitly: sparse/indexer, layer types, hyper-connection,
    # n-gram hashing/partitioning, rotary.
    for field in (
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_kv_heads",
        "indexer_n_heads",
        "index_share_for_mtp_iteration",
    ):
        assert normalized[field] == RVN_CONFIG[field]
    assert normalized["layer_types"] == RVN_CONFIG["layer_types"]
    assert normalized["full_attention_interval"] == 4
    assert normalized["hc_count"] == 4 and normalized["hc_lowrank"] == 320
    assert normalized["ngram_size"] == RVN_CONFIG["ngram_size"]
    assert normalized["heads_per_ngram"] == RVN_CONFIG["heads_per_ngram"]
    assert normalized["split_ngram_parts"] == 128
    assert normalized["seed"] == 1234
    assert normalized["rope_parameters"] == RVN_CONFIG["rope_parameters"]


def test_rvn_detection_is_explicit_not_defaulted(adapter):
    # Architecture-only or model_type-only matches must NOT be treated as RVN.
    assert not adapter.is_rvn_text_model(
        ["Qwen4ExpForCausalLM"], "qwen4_exp"
    )
    assert not adapter.is_rvn_text_model(
        ["Qwen4ExpForConditionalGeneration"], "qwen4_exp_text"
    )
    assert not adapter.is_rvn_text_model([], "qwen4_exp_text")
    assert not adapter.is_rvn_text_model(None, None)


def test_rvn_checked_wrapper_rejects_vision_and_mtp(adapter):
    bad_vision = dict(RVN_CONFIG, vision_config={"depth": 27})
    with pytest.raises(ValueError):
        adapter.assert_rvn_text_config(bad_vision)
    bad_mtp = dict(RVN_CONFIG, mtp_num_hidden_layers=1)
    with pytest.raises(ValueError):
        adapter.assert_rvn_text_config(bad_mtp)
    with pytest.raises(ValueError):
        adapter.normalize_text_config(bad_mtp)
    with pytest.raises(ValueError):
        adapter.assert_rvn_text_config(LIL_CONFIG)  # multimodal is not RVN


def test_lil_multimodal_path_unchanged(adapter):
    import copy

    baseline = copy.deepcopy(LIL_CONFIG)
    normalized = adapter.normalize_text_config(LIL_CONFIG)
    # LIL config passes through byte-for-byte (baseline values).
    assert normalized == baseline
    assert normalized is not baseline
    assert normalized["vision_config"] == LIL_CONFIG["vision_config"]
    assert normalized["text_config"]["mtp_num_hidden_layers"] == 1
    assert not adapter.is_rvn_text_model(
        LIL_CONFIG["architectures"], LIL_CONFIG["model_type"]
    )


# ---------------------------------------------------------------------------
# Weight-name mapping (text prefixes only, shared claim path)
# ---------------------------------------------------------------------------

_PLE_SHARD_W = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"
)
_PLE_SHARD_S = _PLE_SHARD_W.replace(".weight", ".weight_scale")
_PLE_BUFFER = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"


def test_text_weight_prefix_mapping(adapter):
    assert adapter.map_text_weight_prefix(_PLE_SHARD_W) == _PLE_SHARD_W.replace(
        "model.language_model.", "model."
    )
    assert adapter.map_text_weight_prefix("lm_head.weight") == "lm_head.weight"
    assert (
        adapter.map_text_weight_prefix("model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_ple_shard_names_are_claimed(adapter):
    assert adapter.claim_text_weight_name(_PLE_SHARD_W) == "ple_shard"
    assert adapter.claim_text_weight_name(_PLE_SHARD_S) == "ple_shard"
    match = adapter.ple_shard_match(_PLE_SHARD_W)
    assert match is not None and match.group(1) == "3" and match.group(2) == "weight"
    assert adapter.claim_text_weight_name(_PLE_BUFFER) == "ple_buffer"
    assert (
        adapter.claim_text_weight_name(
            "model.language_model.layers.1.ple.ple_embedding.hashstats_combined"
        )
        == "ple_buffer"
    )
    assert (
        adapter.claim_text_weight_name(
            "model.language_model.layers.0.attn_hyper_connection.alpha"
        )
        == "text"
    )
    # The loader claims only canonical ids: shard_03 matches the shape but stays
    # unclaimed by the PLE path and falls through to default handling.
    malformed = _PLE_SHARD_W.replace("shard_3", "shard_03")
    assert adapter.ple_shard_match(malformed) is not None
    assert not adapter.ple_shard_is_canonical(adapter.ple_shard_match(malformed))
    assert adapter.claim_text_weight_name(malformed) == "text"


def test_visual_and_mtp_names_rejected_for_text_path(adapter):
    with pytest.raises(ValueError):
        adapter.claim_text_weight_name(
            "model.language_model.visual.blocks.0.attn.qkv.weight"
        )
    with pytest.raises(ValueError):
        adapter.reject_non_text_weight_name("model.mtp.0.gating.weight")
    with pytest.raises(ValueError):
        adapter.reject_non_text_weight_name(
            "model.visual.patch_embed.proj.weight"
        )
    # Text names pass through the same gate untouched.
    adapter.reject_non_text_weight_name(_PLE_SHARD_W)


def test_ple_global_scale_key_keeps_llava_form_lookup(adapter):
    assert (
        adapter.ple_global_scale_ckpt_key("model.layers.1.ple.ple_embedding")
        == "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2"
    )


# ---------------------------------------------------------------------------
# Patched tree structure (AST): registration, single loader, no visual
# ---------------------------------------------------------------------------


def test_entry_class_registration_via_ast(tree, adapter):
    model_rel = "python/sglang/srt/models/qwen4_exp.py"
    names = _entry_class_names(tree, model_rel)
    assert adapter.MULTIMODAL_ARCHITECTURE in names
    assert adapter.RVN_TEXT_ARCHITECTURE in names
    # MTP draft entry name must match the helper constant (no literal drift).
    mtp_rel = "python/sglang/srt/models/qwen4_exp_mtp.py"
    preimage_mtp = _preimage_root() / "srt/models/qwen4_exp_mtp.py"
    if not (tree / mtp_rel).exists():
        shutil.copyfile(preimage_mtp, tree / mtp_rel)
    mtp_names = _entry_class_names(tree, mtp_rel)
    assert mtp_names == [adapter.MTP_DRAFT_ARCHITECTURE]


def test_both_entries_share_one_loader(tree):
    model_rel = "python/sglang/srt/models/qwen4_exp.py"
    module = _parse(tree, model_rel)
    mixin_defs = [
        n.name
        for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpWeightLoadMixin"
    ]
    assert mixin_defs == ["Qwen4ExpWeightLoadMixin"]
    shared = [
        n.name
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef) and n.name == "load_qwen4_exp_weights"
    ]
    assert len(shared) == 1, "PLE/load path must exist exactly once"
    for cls_name, text_only in (
        ("Qwen4ExpForConditionalGeneration", False),
        ("Qwen4ExpForCausalLM", True),
    ):
        wrapper = _function(tree, model_rel, cls_name, "load_weights")
        calls = [
            n
            for n in ast.walk(wrapper)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load_qwen4_exp_weights"
        ]
        assert len(calls) == 1, cls_name
        (kw,) = calls[0].keywords
        assert kw.arg == "text_only" and isinstance(kw.value, ast.Constant)
        assert kw.value.value is text_only, cls_name


def test_text_adapter_class_builds_no_visual_and_loads_text_only(tree):
    model_rel = "python/sglang/srt/models/qwen4_exp.py"
    init = _function(tree, model_rel, "Qwen4ExpForCausalLM", "__init__")
    visual_assigns = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "visual" for t in n.targets
        )
    ]
    assert visual_assigns, "Qwen4ExpForCausalLM must set self.visual"
    for assign in visual_assigns:
        assert isinstance(assign.value, ast.Constant) and assign.value.value is None
    # Detection gates the class before anything is built.
    gate = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "assert_rvn_text_config"
    ]
    assert gate, "checked wrapper must gate on explicit RVN detection"
    # Shared loader consumes the helper's single prefix/PLE-claim rules.
    loader = [
        n
        for n in ast.walk(_parse(tree, model_rel))
        if isinstance(n, ast.FunctionDef) and n.name == "load_qwen4_exp_weights"
    ][0]
    used = {
        n.func.id
        for n in ast.walk(loader)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    used |= {
        n.func.value.id
        for n in ast.walk(loader)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
    }
    assert {"map_text_weight_prefix", "reject_non_text_weight_name"} <= used
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "search" and isinstance(n.value, ast.Name) and n.value.id == "PLE_SHARD_RE"
        for n in ast.walk(loader)
    ), "shard claim must use the shared PLE_SHARD_RE rule"


def test_config_registry_resolves_rvn_flat_model_type(tree):
    src = (tree / "python/sglang/srt/utils/hf_transformers/common.py").read_text()
    assert '_CONFIG_REGISTRY["qwen4_exp_text"] = Qwen4ExpTextConfig' in src


def test_patched_tree_compiles(tree):
    for rel in (
        "python/sglang/srt/models/qwen4_exp.py",
        "python/sglang/srt/models/qwen4_exp_text_adapter.py",
        "python/sglang/srt/utils/hf_transformers/common.py",
    ):
        py_compile.compile(
            str(tree / rel), cfile=str(tree / f"{Path(rel).name}.pyc"), doraise=True
        )

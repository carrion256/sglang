"""WP2 integration seam: patches/0050-rvn-ple-hooksite.patch.

The RVN text load path (``Qwen4ExpWeightLoadMixin.load_qwen4_exp_weights``
with ``text_only=True``) must actually CALL the 0049 packed-PLE loader hook
``rvn_ple_storage_for_checkpoint`` and account the manifest-replaced PLE
table tensors, with zero delta for the LIL multimodal (``text_only=False``)
path. Conventions (repo style):
- Code under test is the patched tree's single files, loaded with
  ``importlib``/``ast`` extraction; never ``import sglang`` host-scope.
- ``RVN_PLE_TREE`` points at the root of the tree 0047+0048+0049+0050 were
  applied to with ``-p1`` (the throwaway container does exactly this); the
  module skips when unset so ``pytest tests/`` stays green on unpatched trees.
- The 0050-preimage baseline (0047+0048+0049) is rebuilt inside the test by
  reverse-applying patches/0050 onto a scratch copy of the stacked tree, and
  behavior is compared variant-vs-variant like test_rvn_w4a16_dispatch does.
- Fixtures are synthetic checkpoints under ``tmp_path`` reusing the schema
  builders of test_rvn_ple_loader.py; the real RVN checkpoint is never
  touched.
"""

import ast
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PATCH_0050 = REPO / "patches" / "0050-rvn-ple-hooksite.patch"

_MODEL_REL = Path("python/sglang/srt/models/qwen4_exp.py")
_ADAPTER_REL = Path("python/sglang/srt/models/qwen4_exp_text_adapter.py")
_STORAGE_REL = Path("python/sglang/srt/models/rvn_ple_storage.py")
_PLE_REL = Path("python/sglang/srt/models/packed_ple.py")
_WU_REL = Path("python/sglang/srt/model_loader/weight_utils.py")

MOD_PREFIX = "model.layers.1.ple.ple_embedding"
# RVN checkpoints are LLaVA-form (RvnTextAdapter: live-verified), so the
# manifest's source_tensor names carry the producer prefix and must flow
# through the shared map_text_weight_prefix rule.
SOURCE_TENSOR = ("model.language_model.layers.1.ple.ple_embedding"
                 ".ngram_embedding.weight")
TABLE_PARAM = f"{MOD_PREFIX}.ngram_embedding.weight"
ROWS = 8
COLS = 32


def _tree():
    tree = os.environ.get("RVN_PLE_TREE")
    if tree:
        root = Path(tree)
        assert (root / _STORAGE_REL).is_file(), (
            f"RVN_PLE_TREE={tree} lacks {_STORAGE_REL}: apply "
            "patches/0047..0050 to that tree root with -p1")
        if not (root / _ADAPTER_REL).is_file():
            # A 0049-only (or older) tree is a valid RVN_PLE_TREE for the
            # loader suite but predates this seam: skip loudly instead of
            # erroring at collection (PleLoader2 mixed-tree battery).
            pytest.skip(
                f"RVN_PLE_TREE={tree} predates patch 0047 "
                f"({_ADAPTER_REL} missing); 0050 wiring requires the "
                "0047+0048+0049+0050 stack", allow_module_level=True)
        return root
    if (REPO / "runtime" / _STORAGE_REL).is_file():
        return REPO / "runtime"
    pytest.skip(
        "the WP2 runtime files ship only inside patches/0047..0050; set "
        "RVN_PLE_TREE to the tree the patches were applied to",
        allow_module_level=True)


TREE = _tree()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, TREE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Schema fixture builders: single copy in the WP2 loader test (same tree,
# same checkpoint contract); imported, not forked.
_loader_spec = importlib.util.spec_from_file_location(
    "rvn_ple_loader_fixtures", REPO / "tests" / "rvn_ple" / "test_rvn_ple_loader.py")
FIX = importlib.util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(FIX)

adapter = _load("rvn_ple_wiring_adapter", _ADAPTER_REL)
rvn = _load("rvn_ple_wiring_storage", _STORAGE_REL)
packed_ple = _load("rvn_ple_wiring_packed_ple", _PLE_REL)


def _source_segment(path, name, kind):
    """Exact source text of one top-level def/class from a file."""
    module = ast.parse(Path(path).read_text())
    for node in module.body:
        if isinstance(node, kind) and node.name == name:
            return ast.get_source_segment(Path(path).read_text(), node)
    raise AssertionError(f"{name} not found in {path}")


HOOK_SRC = _source_segment(TREE / _WU_REL, "rvn_ple_storage_for_checkpoint",
                           ast.FunctionDef)


class _Logger:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, fmt, *args):
        self.infos.append(fmt % args if args else fmt)

    def warning(self, fmt, *args):
        self.warnings.append(fmt % args if args else fmt)


class _MixinEnv:
    """Stub environment + fake model for one extracted-mixin variant."""

    def __init__(self, qwen4_exp_path):
        self.logger = _Logger()

        class PinnedHost:  # stub of Qwen4ExpPinnedHostEmbedding
            def __init__(self):
                self._packed_storage = None
                self._packed_fp8_reference = False
                self.tp_size = 1
                self.embedding_dim = COLS
                self.org_vocab_size = ROWS
                self.weight = torch.nn.Parameter(
                    torch.zeros(ROWS, COLS, dtype=torch.bfloat16),
                    requires_grad=False)
                self.shard_indices = SimpleNamespace(
                    org_vocab_start_index=0, org_vocab_end_index=ROWS,
                    padded_org_vocab_start_index=0,
                    padded_org_vocab_end_index=ROWS)

        class Ngram:  # stub of Qwen4ExpNGramEmbedding
            def __init__(self, emb):
                self._ple_source_packed = False
                self.ngram_embedding = emb

        class GatedDeltaNet:  # stub of Qwen3_5GatedDeltaNet
            pass

        self.PinnedHost, self.Ngram, self.GatedDeltaNet = (
            PinnedHost, Ngram, GatedDeltaNet)

        env = {
            "torch": torch, "os": os, "json": json, "math": __import__("math"),
            "logger": self.logger,
            "Iterable": __import__("typing").Iterable,
            "Tuple": __import__("typing").Tuple,
            "Set": __import__("typing").Set,
            "FusedMoE": SimpleNamespace(make_expert_params_mapping=None),
            "default_weight_loader": None,
            "PLE_SHARD_RE": adapter.PLE_SHARD_RE,
            "_PLE_TABLE_MARK": adapter._PLE_TABLE_MARK,
            "map_text_weight_prefix": adapter.map_text_weight_prefix,
            "ple_shard_is_canonical": adapter.ple_shard_is_canonical,
            "reject_non_text_weight_name": adapter.reject_non_text_weight_name,
            "ple_global_scale_ckpt_key": adapter.ple_global_scale_ckpt_key,
            "Qwen4ExpNGramEmbedding": Ngram,
            "Qwen4ExpPinnedHostEmbedding": PinnedHost,
            "Qwen3_5GatedDeltaNet": GatedDeltaNet,
            "get_layer_id": lambda name: None,
            "_PLE_E2M1_LUT": None,
        }
        src = _source_segment(qwen4_exp_path, "Qwen4ExpWeightLoadMixin",
                              ast.ClassDef)
        exec(compile(src, str(qwen4_exp_path), "exec"), env)
        self.mixin = env["Qwen4ExpWeightLoadMixin"]

        # A minimal text-only model: one PLE layer at MOD_PREFIX, no params.
        class FakeModel(self.mixin):
            language_model_only = False
            start_layer, end_layer = 0, 99
            pp_group = SimpleNamespace(is_last_rank=False)

            def __init__(self, config, ple_mods):
                self.config = config
                self._ple_mods = ple_mods

            def named_modules(self):
                return list(self._ple_mods.items())

            def modules(self):
                return [mod for _, mod in self._ple_mods.items()]

            def named_parameters(self, remove_duplicate=False):
                return {}

            def named_buffers(self):
                return {}

            def post_load_weights(self):
                pass

            def _log_weight_dtype_census(self):
                pass

        self.FakeModel = FakeModel

    def model(self, ckpt_root, *, source_packed=False, with_storage=True):
        emb = self.PinnedHost()
        if with_storage:
            emb._packed_storage = packed_ple.PackedPLEStorage(
                ROWS, COLS, pin_memory=False)
        else:
            emb.weight = torch.nn.Parameter(
                torch.zeros(ROWS, COLS, dtype=torch.bfloat16),
                requires_grad=False)
        mod = self.Ngram(emb)
        mod._ple_source_packed = source_packed
        return self.FakeModel(
            SimpleNamespace(_name_or_path=str(ckpt_root), num_experts=None,
                            tie_word_embeddings=False, split_ngram_parts=2),
            {MOD_PREFIX: mod})


@contextlib.contextmanager
def _sglang_stubs(model_path):
    """Register just enough of ``sglang.*`` for the lazy hook imports, using
    the REAL tree modules; monkeypatch restores sys.modules afterwards."""
    hook_ns = {}
    exec(compile(HOOK_SRC, str(TREE / _WU_REL), "exec"),
         {"Optional": __import__("typing").Optional}, hook_ns)
    weight_utils = types.ModuleType("sglang.srt.model_loader.weight_utils")
    weight_utils.rvn_ple_storage_for_checkpoint = hook_ns[
        "rvn_ple_storage_for_checkpoint"]
    stubs = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.model_loader": types.ModuleType("sglang.srt.model_loader"),
        "sglang.srt.model_loader.weight_utils": weight_utils,
        "sglang.srt.models": types.ModuleType("sglang.srt.models"),
        "sglang.srt.models.rvn_ple_storage": rvn,
        "sglang.srt.models.packed_ple": packed_ple,
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
    }
    stubs["sglang"].__path__ = []
    for name in ("sglang.srt", "sglang.srt.model_loader", "sglang.srt.models"):
        stubs[name].__path__ = []
    stubs["sglang.srt"].server_args = stubs["sglang.srt.server_args"]
    stubs["sglang.srt.server_args"].get_global_server_args = (
        lambda: SimpleNamespace(model_path=model_path))
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield weight_utils
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _checkpoint(tmp_path, *, source_tensor=SOURCE_TENSOR, rows_a=FIX.ROWS_A,
                rows_b=FIX.ROWS_B, amax=FIX.AMAX_TIE, manifest=True):
    """Synthetic RVN manifest checkpoint: valid parts + ple_storage.json with
    LLaVA-form source_tensor names."""
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    packed = torch.cat([FIX._grid_rows(rows_a), FIX._grid_rows(rows_b, seed=8)])
    scales = torch.cat([FIX._scale_rows(rows_a),
                        FIX._scale_rows(rows_b, seed=12)])
    parts = [FIX._write_part(root, 0, packed[:rows_a], scales[:rows_a]),
             FIX._write_part(root, 1, packed[rows_a:], scales[rows_a:])]
    manifest_dict = FIX._manifest(parts, amax=amax, logical_rows=rows_a + rows_b)
    for entry in manifest_dict["table"]["partitioning"]:
        entry["source_tensor"] = source_tensor
    if manifest is not True:
        (root / "ple_storage.json").write_bytes(manifest)
    else:
        (root / "ple_storage.json").write_text(json.dumps(manifest_dict))
    return root, packed, scales


def _revert_later_stacked_patches(tree):
    """Reverse-apply the profile patches stacked after 0050 that touch the two
    files this harness copies. 0054's reconstruction gate lands *inside* the
    block 0050 adds, so a bare reverse of 0050 would no longer find its
    post-image; undoing the later stacked hunks first restores it. Patches
    this tree does not carry fail ``--check`` and are skipped."""
    include = [f"--include={rel}" for rel in (_MODEL_REL, _WU_REL)]
    for path in sorted((REPO / "patches").glob("*.patch")):
        if path.name <= PATCH_0050.name:
            continue
        argv = ["git", "apply", "-R", "-p1", *include, str(path)]
        probe = subprocess.run(argv + ["--check"], cwd=tree,
                               capture_output=True)
        if probe.returncode == 0:
            subprocess.run(argv, cwd=tree, check=True, capture_output=True)


def _stacked_and_baseline(tmp_path):
    """(stacked, 0050-reverted) copies of qwen4_exp.py; the baseline is the
    exact 0050 preimage, rebuilt by ``git apply -R`` of the repo patch (after
    any patch stacked on top of it, see ``_revert_later_stacked_patches``)."""
    root = tmp_path / "rvn0050-compare"
    paths = {}
    for variant in ("stacked", "preimage"):
        tree = root / variant
        for rel in (_MODEL_REL, _WU_REL):
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(TREE / rel, dst)
        if variant == "preimage":
            _revert_later_stacked_patches(tree)
            subprocess.run(
                ["git", "apply", "-R", "-p1", str(PATCH_0050)],
                cwd=tree, check=True, capture_output=True)
        paths[variant] = tree / _MODEL_REL
    return paths["stacked"], paths["preimage"]


# ------------------------------------------------------------------- tests


def test_rvn_text_load_with_manifest_calls_hook_and_fills_packed_storage(tmp_path):
    root, packed, scales = _checkpoint(tmp_path)
    env = _MixinEnv(TREE / _MODEL_REL)
    model = env.model(root)
    bf16_table = model._ple_mods[MOD_PREFIX].ngram_embedding.weight
    real_copy_ = torch.Tensor.copy_

    def guarded_copy_(self, other, *args, **kwargs):
        # Catch writes to the BF16 host table itself AND to any view/slice of
        # it (the legacy path writes rows through emb.weight.data[...]).
        if (self.untyped_storage().data_ptr()
                == bf16_table.untyped_storage().data_ptr()):
            raise AssertionError("BF16 host table was assigned on the manifest path")
        return real_copy_(self, other, *args, **kwargs)

    torch.Tensor.copy_ = guarded_copy_
    try:
        with _sglang_stubs(str(root)) as weight_utils:
            calls = []
            real_hook = weight_utils.rvn_ple_storage_for_checkpoint

            def spy(*args, **kwargs):
                result = real_hook(*args, **kwargs)
                calls.append((args, kwargs, result))
                return result

            weight_utils.rvn_ple_storage_for_checkpoint = spy
            loaded = model.load_qwen4_exp_weights(
                [("model.language_model.embed_tokens.weight",
                  torch.zeros(2, 2, dtype=torch.bfloat16)),
                 (SOURCE_TENSOR,
                  torch.zeros(ROWS, COLS, dtype=torch.bfloat16))],
                text_only=True)
    finally:
        torch.Tensor.copy_ = real_copy_

    emb = model._ple_mods[MOD_PREFIX].ngram_embedding
    storage = emb._packed_storage
    # The hook was consulted exactly once, filling the embedding's own slot
    # over this rank's window (no second storage, no BF16 table).
    assert len(calls) == 1
    (args, kwargs, returned) = calls[0]
    assert args == (str(root),)
    assert kwargs["storage"] is storage and returned is storage
    assert kwargs["tp_start"] == 0 and kwargs["tp_end"] == ROWS
    # The slot holds exactly the object the hook returned.
    assert emb._packed_storage is returned
    assert emb._rvn_ple_manifest_loaded is True
    assert storage.global_scale == FIX._f32(FIX.AMAX_TIE / (6.0 * FIX.E4M3_MAX))
    assert torch.equal(storage.weight, packed)
    assert torch.equal(storage.scales.view(torch.uint8), scales.view(torch.uint8))
    # Packed-replaced table tensor accounted; the covered checkpoint tensor was
    # consumed, and the untouched BF16 host table proves no expansion.
    assert TABLE_PARAM in loaded
    assert adapter.map_text_weight_prefix(SOURCE_TENSOR) in loaded
    assert torch.equal(bf16_table, torch.zeros_like(bf16_table))
    assert any("ple_storage.json" in line for line in env.logger.infos)


def test_hook_assembles_when_slot_missing(tmp_path):
    """No pre-existing packed slot (packed host embedding built without the
    nvfp4 flag): the seam constructs the storage instead of BF16-expanding."""
    root, packed, _ = _checkpoint(tmp_path)
    env = _MixinEnv(TREE / _MODEL_REL)
    model = env.model(root, with_storage=False)
    emb = model._ple_mods[MOD_PREFIX].ngram_embedding
    with _sglang_stubs(str(root)):
        model.load_qwen4_exp_weights([], text_only=True)
    assert torch.equal(emb._packed_storage.weight, packed)


def test_no_manifest_keeps_legacy_path_and_never_calls_hook(tmp_path):
    """Preimage-vs-stacked: without ple_storage.json the 0050 code must not
    change the legacy load, and the hook must stay unconsulted — for the RVN
    text entry too. LIL (text_only=False) never consults even WITH a manifest."""
    root = tmp_path / "lil"
    root.mkdir()
    stacked_path, preimage_path = _stacked_and_baseline(tmp_path)
    with open(root / "model.safetensors", "wb") as handle:
        handle.write(b"untouched")

    stream = [(f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
               f".shard_0.weight",
               torch.full((4, COLS), 0.25, dtype=torch.bfloat16))]
    results = {}
    for variant, path in (("stacked", stacked_path), ("preimage", preimage_path)):
        env = _MixinEnv(path)
        model = env.model(root, with_storage=False)
        with _sglang_stubs(str(root)) as weight_utils:
            weight_utils.rvn_ple_storage_for_checkpoint = (
                lambda *a, **k: pytest.fail(
                    "packed-PLE hook consulted without a manifest"))
            loaded = model.load_qwen4_exp_weights(
                list(stream), text_only=True)
        emb = model._ple_mods[MOD_PREFIX].ngram_embedding
        results[variant] = (
            loaded, emb.weight.data.clone(), list(env.logger.infos),
            getattr(emb, "_packed_storage", None))
    assert results["stacked"][0] == results["preimage"][0] == {TABLE_PARAM}
    assert torch.equal(results["stacked"][1], results["preimage"][1])
    assert torch.equal(results["stacked"][1][0:4],
                       torch.full((4, COLS), 0.25, dtype=torch.bfloat16))
    assert results["stacked"][2] == results["preimage"][2]
    assert results["stacked"][3] is None is results["preimage"][3]
    assert not (root / "ple_storage.json").exists()
    assert (root / "model.safetensors").read_bytes() == b"untouched"


def test_lil_multimodal_path_never_consults_manifest(tmp_path):
    """Zero-delta on the LIL arch: with text_only=False the hook and the
    manifest are never consulted, even when the checkpoint IS manifest-driven."""
    root, _, _ = _checkpoint(tmp_path)
    stacked_path, preimage_path = _stacked_and_baseline(tmp_path)
    stream = [("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
               ".shard_0.weight",
               torch.full((4, COLS), 0.5, dtype=torch.bfloat16))]
    results = {}
    for variant, path in (("stacked", stacked_path), ("preimage", preimage_path)):
        env = _MixinEnv(path)
        model = env.model(root, with_storage=False)
        with _sglang_stubs(str(root)) as weight_utils:
            weight_utils.rvn_ple_storage_for_checkpoint = (
                lambda *a, **k: pytest.fail(
                    "packed-PLE hook consulted on the LIL text_only=False path"))
            loaded = model.load_qwen4_exp_weights(list(stream), text_only=False)
        emb = model._ple_mods[MOD_PREFIX].ngram_embedding
        results[variant] = (loaded, emb.weight.data.clone(),
                            emb._rvn_ple_manifest_loaded
                            if hasattr(emb, "_rvn_ple_manifest_loaded") else None)
    assert results["stacked"][0] == results["preimage"][0] == {TABLE_PARAM}
    assert torch.equal(results["stacked"][1], results["preimage"][1])
    assert results["stacked"][2] is None  # manifest flag never set for LIL


def test_invalid_manifest_raises_through_load_path(tmp_path):
    for bad in ("{ not json", json.dumps({"format_version": 2})):
        root = tmp_path / ("ckpt-" + str(abs(hash(bad))))
        root.mkdir()
        (root / "ple_storage.json").write_text(bad)
        env = _MixinEnv(TREE / _MODEL_REL)
        model = env.model(root)
        with _sglang_stubs(str(root)):
            with pytest.raises(ValueError):
                model.load_qwen4_exp_weights([], text_only=True)
        # No fallback: the model never reaches the legacy tail.
        assert not env.logger.infos or all(
            "ple_storage.json" not in line for line in env.logger.infos)


def test_manifest_coverage_hole_is_rejected_at_load(tmp_path):
    # (1) A manifest whose rows do not reach the model's vocab window.
    root = tmp_path / "hole-rows"
    root.mkdir()
    packed = FIX._grid_rows(FIX.ROWS_A)
    scales = FIX._scale_rows(FIX.ROWS_A)
    parts = [FIX._write_part(root, 0, packed, scales)]
    manifest = FIX._manifest(parts, logical_rows=FIX.ROWS_A)
    for entry in manifest["table"]["partitioning"]:
        entry["source_tensor"] = SOURCE_TENSOR
    (root / "ple_storage.json").write_text(json.dumps(manifest))
    env = _MixinEnv(TREE / _MODEL_REL)
    with _sglang_stubs(str(root)):
        with pytest.raises(ValueError, match="TP window|incomplete"):
            env.model(root).load_qwen4_exp_weights([], text_only=True)

    # (2) A manifest naming a PLE module the model does not have: the manifest
    # itself stays schema-valid, only the coverage names point off-model.
    root2, _, _ = _checkpoint(
        tmp_path,
        source_tensor="model.language_model.layers.9.ple.ple_embedding"
                      ".ngram_embedding.weight")
    env2 = _MixinEnv(TREE / _MODEL_REL)
    with _sglang_stubs(str(root2)):
        with pytest.raises(ValueError, match="coverage hole"):
            env2.model(root2).load_qwen4_exp_weights([], text_only=True)


def test_uncovered_ple_table_tensor_fails_loudly(tmp_path):
    """Strict accounting: a PLE table tensor the manifest does not cover must
    never be expanded to BF16; it fails at the load path instead."""
    root, _, _ = _checkpoint(tmp_path)
    env = _MixinEnv(TREE / _MODEL_REL)
    model = env.model(root)
    stray = ("model.language_model.layers.1.ple.ple_embedding"
             ".ngram_embedding.shard_7.weight")
    with _sglang_stubs(str(root)):
        with pytest.raises(ValueError, match="not covered by ple_storage"):
            model.load_qwen4_exp_weights(
                [(stray, torch.zeros(1, COLS, dtype=torch.uint8))],
                text_only=True)

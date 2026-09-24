"""Serve-time binding of the manifest's packed-PLE reconstruction mode
(patch 0054, review finding P1#5).

``ple_storage.json`` declares ``encoding.reconstruction``, and the frozen
schema (docs/rvn-ple-storage-schema.md section 1) scopes RVN v1 to
``bf16_direct`` while keeping the legacy ``fp8_roundtrip`` gather mode merely
*selectable*. Before 0054 the declaration was parsed, validated and then
ignored: the mode actually used by every PLE lookup was
``Qwen4ExpPinnedHostEmbedding._packed_fp8_reference``, derived by the
embedding from ``SGLANG_PLE_PACKED_FP8_REFERENCE``, and every documented
packed-serve environment in this repo sets that flag. Serving a
``bf16_direct`` candidate with the production environment therefore rounded
every PLE lookup through the out-of-scope E4M3 round-trip, silently.

Conventions (repo style, as in test_rvn_ple_wiring.py):
- The code under test is the stacked tree's own ``qwen4_exp.py``, exec'd
  through the 0050 wiring harness; never ``import sglang`` host-scope.
- The 0054 preimage is rebuilt inside the test by reverse-applying
  patches/0054 onto a scratch copy, so each test pins hole-and-fix in one
  run. A tree that predates 0054 skips loudly (mixed-tree battery).
- The embedding's gather-kernel mode is *not* re-declared here: it is
  obtained by exec'ing the deployed assignment from the tree itself, so the
  environment semantics under test are the real ones.
- Fixtures are synthetic checkpoints under ``tmp_path``; the real RVN
  checkpoint is never touched.
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PATCH_0054 = REPO / "patches" / "0054-rvn-ple-recon-mode-gate.patch"
_MODEL_REL = Path("python/sglang/srt/models/qwen4_exp.py")
ENV_FLAG = "SGLANG_PLE_PACKED_FP8_REFERENCE"

# Harness + schema fixture builders: imported from the 0050 wiring suite and
# from the loader suite through it, never forked.
_wiring_spec = importlib.util.spec_from_file_location(
    "rvn_ple_recon_mode_wiring",
    REPO / "tests" / "rvn_ple" / "test_rvn_ple_wiring.py")
W = importlib.util.module_from_spec(_wiring_spec)
_wiring_spec.loader.exec_module(W)

FIX = W.FIX
rvn = W.rvn
packed_ple = W.packed_ple
TREE = W.TREE
MOD_PREFIX = W.MOD_PREFIX
ROWS = W.ROWS
COLS = W.COLS


def _pinned_host_mode_assignment(path):
    """Source of the deployed statement that picks the packed PLE gather
    kernel's fp8-reference mode (``self._packed_fp8_reference = ...``, added
    by patches/0010). The gate binds to exactly this attribute, so the suite
    drives it through the real derivation instead of restating it."""
    src = Path(path).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == (
                "Qwen4ExpPinnedHostEmbedding"):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(target, ast.Attribute)
                        and target.attr == "_packed_fp8_reference"
                        for target in stmt.targets):
                    segment = ast.get_source_segment(src, stmt)
                    assert segment is not None
                    return segment
    raise AssertionError(
        f"Qwen4ExpPinnedHostEmbedding never derives _packed_fp8_reference "
        f"in {path}")


MODE_ASSIGNMENT = _pinned_host_mode_assignment(TREE / _MODEL_REL)


def _effective_mode():
    """The gather-kernel mode the deployed embedding would use right now,
    computed by running the tree's own assignment against the environment."""
    holder = SimpleNamespace()
    exec(compile(MODE_ASSIGNMENT, str(TREE / _MODEL_REL), "exec"),
         {"os": os, "self": holder})
    return holder._packed_fp8_reference


@pytest.fixture
def gate(tmp_path):
    """``(gated, ungated)`` copies of ``qwen4_exp.py``: the stacked tree and
    the exact 0054 preimage, rebuilt with ``git apply -R``."""
    root = tmp_path / "rvn0054-compare"
    paths = {}
    for variant in ("gated", "ungated"):
        tree = root / variant
        dst = tree / _MODEL_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TREE / _MODEL_REL, dst)
        if variant == "ungated":
            probe = subprocess.run(
                ["git", "apply", "-R", "-p1", "--check", str(PATCH_0054)],
                cwd=tree, capture_output=True)
            if probe.returncode != 0:
                pytest.skip(
                    f"RVN_PLE_TREE={TREE} predates "
                    "patches/0054-rvn-ple-recon-mode-gate.patch "
                    f"({probe.stderr.decode().strip()}); the mode gate is "
                    "only enforceable once it is in the applied stack")
            subprocess.run(
                ["git", "apply", "-R", "-p1", str(PATCH_0054)],
                cwd=tree, check=True, capture_output=True)
        paths[variant] = dst
    return paths["gated"], paths["ungated"]


def _checkpoint(tmp_path, *, reconstruction="bf16_direct"):
    """Valid manifest checkpoint (0050 wiring fixture shape) declaring
    ``reconstruction``, over the 0049 loader's own schema builders."""
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    packed = torch.cat([FIX._grid_rows(FIX.ROWS_A),
                        FIX._grid_rows(FIX.ROWS_B, seed=8)])
    scales = torch.cat([FIX._scale_rows(FIX.ROWS_A),
                        FIX._scale_rows(FIX.ROWS_B, seed=12)])
    parts = [FIX._write_part(root, 0, packed[:FIX.ROWS_A], scales[:FIX.ROWS_A]),
             FIX._write_part(root, 1, packed[FIX.ROWS_A:], scales[FIX.ROWS_A:])]
    (root / "ple_storage.json").write_text(json.dumps(
        FIX._manifest(parts, reconstruction=reconstruction)))
    return root, packed, scales


def _load(model_path, root, monkeypatch, env_value):
    """One RVN text load under ``SGLANG_PLE_PACKED_FP8_REFERENCE=env_value``
    (``None`` = unset), with the embedding's kernel mode taken from the
    deployed derivation. Returns ``(env, emb, loaded, mode)``."""
    monkeypatch.delenv(ENV_FLAG, raising=False)
    if env_value is not None:
        monkeypatch.setenv(ENV_FLAG, env_value)
    mode = _effective_mode()
    env = W._MixinEnv(model_path)
    model = env.model(root)
    emb = model._ple_mods[MOD_PREFIX].ngram_embedding
    emb._packed_fp8_reference = mode
    with W._sglang_stubs(str(root)):
        loaded = model.load_qwen4_exp_weights([], text_only=True)
    return env, emb, loaded, mode


# ------------------------------------------------------------------- tests


def test_bf16_direct_manifest_with_fp8_kernel_mode_fails_closed(gate, tmp_path,
                                                                monkeypatch):
    """Red before / green after: pre-0054 the contradiction loaded silently;
    the gated tree refuses it, and the error names both values."""
    gated, ungated = gate
    root, packed, _ = _checkpoint(tmp_path, reconstruction="bf16_direct")

    _, emb, _, mode = _load(ungated, root, monkeypatch, "1")
    assert mode is True  # the environment really does request the round-trip
    # The hole: the table loaded anyway, under the mode the manifest forbids.
    assert emb._rvn_ple_manifest_loaded is True
    assert torch.equal(emb._packed_storage.weight, packed)

    with pytest.raises(ValueError) as excinfo:
        _load(gated, root, monkeypatch, "1")
    message = str(excinfo.value)
    # Both sides named: the manifest's declaration and the kernel's mode.
    assert "encoding.reconstruction='bf16_direct'" in message
    assert "fp8_reference=False" in message
    assert "fp8_reference=True" in message
    assert ENV_FLAG in message
    assert "fp8_roundtrip" in message  # and how to satisfy it legitimately


def test_contradiction_refuses_before_any_part_is_read(gate, tmp_path,
                                                       monkeypatch):
    """Fail closed means fail *early*: the rank's host table is untouched and
    no manifest-load was logged, so no partial PLE row can survive the abort."""
    gated, _ = gate
    root, _, _ = _checkpoint(tmp_path, reconstruction="fp8_roundtrip")
    env, emb, _, mode = _load(gated, root, monkeypatch, None)
    assert mode is False
    storage = emb._packed_storage
    assert storage.global_scale is None
    assert torch.equal(storage.weight, torch.zeros_like(storage.weight))
    assert not getattr(emb, "_rvn_ple_manifest_loaded", False)
    assert not [line for line in env.logger.infos if "rvn-ple" in line]


@pytest.mark.parametrize("reconstruction,env_value", [
    ("bf16_direct", None),      # the documented RVN serve: flag unset
    ("bf16_direct", "0"),
    ("bf16_direct", "false"),   # not "1": off per the deployed derivation
    ("bf16_direct", "true"),    # ditto -- truthy spellings are not "1"
    ("fp8_roundtrip", "1"),     # the legacy mode stays selectable, matching
])
def test_matching_modes_load_and_log_the_bound_mode(gate, tmp_path,
                                                    monkeypatch, reconstruction,
                                                    env_value):
    """The gate binds the two modes; it never bans one of them."""
    gated, _ = gate
    root, packed, scales = _checkpoint(tmp_path, reconstruction=reconstruction)
    env, emb, loaded, mode = _load(gated, root, monkeypatch, env_value)
    assert mode is (reconstruction == "fp8_roundtrip")
    assert emb._packed_fp8_reference is mode
    assert emb._rvn_ple_manifest_loaded is True
    assert f"{MOD_PREFIX}.ngram_embedding.weight" in loaded
    storage = emb._packed_storage
    assert torch.equal(storage.weight, packed)
    assert torch.equal(storage.scales.view(torch.uint8),
                       scales.view(torch.uint8))
    assert any(f"reconstruction={reconstruction}" in line
               for line in env.logger.infos)


@pytest.mark.parametrize("reconstruction,env_value", [
    ("bf16_direct", "1"),        # production packed-serve env vs v1 manifest
    ("bf16_direct", "1\n"),      # only the exact "1" requests the round-trip
    ("fp8_roundtrip", None),     # legacy manifest served with the flag unset
    ("fp8_roundtrip", "0"),
])
def test_contradicting_modes_are_refused(gate, tmp_path, monkeypatch,
                                         reconstruction, env_value):
    gated, _ = gate
    root, _, _ = _checkpoint(tmp_path, reconstruction=reconstruction)
    with pytest.raises(ValueError) as excinfo:
        _load(gated, root, monkeypatch, env_value)
    message = str(excinfo.value)
    assert f"encoding.reconstruction='{reconstruction}'" in message
    assert ENV_FLAG in message
    assert "ple_storage manifest reconstruction mismatch" in message


def test_mode_gate_is_scoped_to_manifest_checkpoints(gate, tmp_path,
                                                     monkeypatch):
    """No manifest means no declaration to bind, so the legacy LIL serve must
    keep running the round-trip with the flag set -- zero delta, gated or not."""
    gated, ungated = gate
    root = tmp_path / "no-manifest"
    root.mkdir()
    results = {}
    for variant, path in (("gated", gated), ("ungated", ungated)):
        env, emb, loaded, mode = _load(path, root, monkeypatch, "1")
        results[variant] = (loaded, mode, emb._packed_storage,
                            list(env.logger.infos))
    assert results["gated"] == results["ungated"]
    assert results["gated"][1] is True  # flag honored, nothing gated
    assert results["gated"][2] is None  # legacy path never consulted the hook
    assert results["gated"][0] == set()


def test_gated_mode_is_what_the_gather_kernel_actually_uses(tmp_path, monkeypatch,
                                                            gate):
    """The gated boolean is numerically load-bearing: on the same assembled
    table, the direct and round-trip gather modes disagree bit-for-bit while
    the mode the gate accepted reproduces schema section 1 exactly."""
    gated, _ = gate
    root, packed, scales = _checkpoint(tmp_path, reconstruction="bf16_direct")
    _, emb, _, _ = _load(gated, root, monkeypatch, None)
    storage = emb._packed_storage
    assert emb._packed_fp8_reference is rvn.load_manifest(
        str(root)).fp8_reference is False

    rows = torch.arange(FIX.LOGICAL_ROWS)
    expected = rvn.dequant_reference(packed[rows], scales[rows],
                                     storage.global_scale,
                                     fp8_reference=emb._packed_fp8_reference)
    direct = rvn.dequant_reference(packed[rows], scales[rows],
                                   storage.global_scale, fp8_reference=False)
    roundtrip = rvn.dequant_reference(packed[rows], scales[rows],
                                      storage.global_scale, fp8_reference=True)
    assert torch.equal(expected, direct)
    # If this ever compares equal, the gate is guarding nothing.
    assert not torch.equal(direct, roundtrip)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires a free CUDA GPU (production workers hold"
                           " the GPUs)")
def test_gather_kernel_mode_matches_the_bound_manifest(tmp_path):
    """GPU: run the real triton gather with the mode the gate bound for this
    checkpoint and require schema-exact rows; the contradicted mode must not
    reproduce them."""
    root, packed, scales = _checkpoint(tmp_path, reconstruction="bf16_direct")
    manifest = rvn.load_manifest(str(root))
    storage = rvn.load_for_checkpoint(
        str(root), storage=packed_ple.PackedPLEStorage(
            FIX.LOGICAL_ROWS, COLS, pin_memory=False))
    assert manifest.fp8_reference is False

    device = "cuda"
    weight = storage.weight.to(device)
    scales_gpu = storage.scales.to(device)
    ids = torch.arange(FIX.LOGICAL_ROWS, dtype=torch.int64, device=device)

    def gather(fp8_reference):
        out = torch.empty((ids.numel(), COLS), dtype=torch.bfloat16,
                          device=device)
        packed_ple.gather_packed_kernel[(ids.numel(),)](
            weight.data_ptr(), scales_gpu.data_ptr(), ids, out,
            storage.global_scale, COLS, 0, FIX.LOGICAL_ROWS, fp8_reference,
            COLS, enable_fp_fusion=False)
        return out.cpu()

    bound = gather(manifest.fp8_reference)
    torch.testing.assert_close(
        bound, rvn.dequant_reference(packed, scales, storage.global_scale,
                                     fp8_reference=manifest.fp8_reference),
        rtol=0, atol=0)
    assert not torch.equal(bound, gather(not manifest.fp8_reference))

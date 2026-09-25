"""Manifest ``encoder_version`` gate (patch 0056).

``rvn_ple_storage.parse_manifest`` validated ``encoder_version`` as a
non-empty string and nothing else: a checkpoint written by any other
encoder build -- a producer version renamed by a later schema revision,
or a hand-edited ``ple_storage.json`` -- loaded silently and served. The
frozen schema value (docs/rvn-ple-storage-schema.md section 2) is
``rvn-ple-nvfp4-r1``, and the offline verifier
(``tools/rvn_ple/verify.py`` ``ENCODER_VERSION``) already refuses every
other producer version, so the runtime must not bless what the verifier
rejected: the loader now fails closed on any value other than the frozen
one, naming both the observed and the expected literal.

Conventions (repo style, as in test_rvn_ple_recon_mode_gate.py):
- The code under test is the stacked tree's own ``rvn_ple_storage.py``,
  exec'd through importlib; never ``import sglang`` host-scope.
- The 0056 preimage is rebuilt inside the test by reverse-applying
  patches/0056 onto a scratch copy, so each test pins hole-and-fix in
  one run. A tree that predates 0056 skips loudly (mixed-tree battery).
- Fixtures are the loader suite's own manifest builders; the real RVN
  checkpoint is never touched.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PATCH_0056 = REPO / "patches" / "0056-rvn-ple-encoder-version-gate.patch"
_STORAGE_REL = Path("python/sglang/srt/models/rvn_ple_storage.py")
# The frozen schema section 2 value, re-declared here independently of
# both the loader and the verifier, per the tests/rvn_ple convention.
FROZEN = "rvn-ple-nvfp4-r1"

# Harness + schema fixture builders: imported from the loader suite
# (same tree, same checkpoint contract), never forked.
_loader_spec = importlib.util.spec_from_file_location(
    "rvn_ple_encoder_gate_fixtures",
    REPO / "tests" / "rvn_ple" / "test_rvn_ple_loader.py")
FIX = importlib.util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(FIX)

TREE = FIX.TREE


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(tmp_path):
    """``(gated, ungated)`` modules: the stacked tree's loader and the
    exact 0056 preimage, rebuilt with ``git apply -R``."""
    root = tmp_path / "rvn0056-compare"
    mods = {}
    for variant in ("gated", "ungated"):
        tree = root / variant
        dst = tree / _STORAGE_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TREE / _STORAGE_REL, dst)
        if variant == "ungated":
            probe = subprocess.run(
                ["git", "apply", "-R", "-p1", "--check", str(PATCH_0056)],
                cwd=tree, capture_output=True)
            if probe.returncode != 0:
                pytest.skip(
                    f"RVN_PLE_TREE={TREE} predates "
                    "patches/0056-rvn-ple-encoder-version-gate.patch "
                    f"({probe.stderr.decode().strip()}); the encoder gate "
                    "is only enforceable once it is in the applied stack")
            subprocess.run(
                ["git", "apply", "-R", "-p1", str(PATCH_0056)],
                cwd=tree, check=True, capture_output=True)
        mods[variant] = _load(f"rvn_ple_storage_{variant}", dst)
    return mods["gated"], mods["ungated"]


def _raw_manifest(tmp_path, encoder_version):
    """Valid two-part manifest (loader-suite shape) with the top-level
    ``encoder_version`` set to ``encoder_version``."""
    packed = torch.cat([FIX._grid_rows(FIX.ROWS_A),
                        FIX._grid_rows(FIX.ROWS_B, seed=8)])
    scales = torch.cat([FIX._scale_rows(FIX.ROWS_A),
                        FIX._scale_rows(FIX.ROWS_B, seed=12)])
    root = tmp_path / "ckpt"
    root.mkdir(parents=True, exist_ok=True)
    parts = [FIX._write_part(root, 0, packed[:FIX.ROWS_A], scales[:FIX.ROWS_A]),
             FIX._write_part(root, 1, packed[FIX.ROWS_A:], scales[FIX.ROWS_A:])]
    raw = FIX._manifest(parts)
    raw["encoder_version"] = encoder_version
    return raw


# ------------------------------------------------------------------- tests


def test_frozen_encoder_version_still_loads(gate, tmp_path):
    """The gate pins the one legal value; it never bans it."""
    gated, _ = gate
    manifest = gated.parse_manifest(_raw_manifest(tmp_path, FROZEN))
    assert manifest.encoder_version == FROZEN


@pytest.mark.parametrize("version", ["rvn-ple-ncfp4-r1", "rvn-ple-nvfp4-r2"])
def test_pre0056_loader_accepted_other_versions_silently(gate, tmp_path,
                                                         version):
    """Red-before half: the 0056 preimage only checked non-empty-string,
    so a foreign encoder build passed the manifest parse unnoticed."""
    _, ungated = gate
    manifest = ungated.parse_manifest(_raw_manifest(tmp_path, version))
    assert manifest.encoder_version == version


@pytest.mark.parametrize("version", ["rvn-ple-ncfp4-r1", "rvn-ple-nvfp4-r2"])
def test_gated_loader_refuses_every_other_version(gate, tmp_path, version):
    """Green-after half: fail closed, naming observed and frozen values
    so an operator can tell a wrong producer from a corrupt manifest."""
    gated, _ = gate
    with pytest.raises(ValueError, match="encoder_version") as exc:
        gated.parse_manifest(_raw_manifest(tmp_path, version))
    message = str(exc.value)
    assert repr(version) in message  # the observed value
    assert repr(FROZEN) in message   # the frozen schema §2 expectation

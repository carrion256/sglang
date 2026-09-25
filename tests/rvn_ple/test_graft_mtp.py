"""Synthetic tests for tools/rvn_ple/graft_mtp.py and verify.py --graft.

Fixtures are tiny torch+safetensors checkpoints written under ``tmp_path``; the
real checkpoints are never written to. The two real-path tests read index JSONs
only -- the full byte-identity pass over them re-hashes 1.5 GiB twice and is
opt-in through ``RVN_PLE_GRAFT_REAL=1``, so the suite stays seconds-fast.

Conventions (repo style):
- Code under test is loaded with ``importlib`` from this repo's ``tools/rvn_ple``,
  and ``RVN_PLE_TREE`` (the root the RVN series was applied to with -p1) must be
  set, because the graft stamp is only meaningful against the shipped loader's
  frozen ``rvn_mtp_count`` rather than a copy of its arithmetic here. The module
  skips cleanly when it is unset.
- Tampering only ever targets files the graft wrote itself. Every base file in
  ``out/`` is a hardlink of the base checkpoint, so a byte written there would
  corrupt the base in place; breaking the link first is part of those tests.
- Expected digests come from the tensors that were saved, never from the tool's
  own byte offsets.
"""

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
GRAFT_TOOL = REPO / "tools" / "rvn_ple" / "graft_mtp.py"
VERIFY_TOOL = REPO / "tools" / "rvn_ple" / "verify.py"
ADAPTER_REL = Path("python/sglang/srt/models/qwen4_exp_text_adapter.py")

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
GRAFT_MANIFEST_NAME = "mtp_graft.json"
GRAFT_STAMP_KEY = "rvn_mtp_graft"
ENCODER_VERSION = "rvn-mtp-graft-r1"
RVN_ARCH = "Qwen4ExpForCausalLM"
RVN_MODEL_TYPE = "qwen4_exp_text"

GRAFT_CHECK_ORDER = (
    "graft_manifest_schema",
    "graft_completeness",
    "graft_provenance",
    "graft_payload_identity",
    "graft_index",
    "graft_config_stamp",
    "graft_files",
)

# The real graft, for the read-only real-path tests.
REAL_SOURCE = Path("/models/qwen38-flash-next")
REAL_BASE = Path("/models/rvn-qwen38-ple-nvfp4")
REAL_OUT = Path("/models/rvn-qwen38-ple-nvfp4-mtp")
REAL_MTP_TENSORS = 4637


def _tree_root():
    tree = os.environ.get("RVN_PLE_TREE")
    if not tree:
        pytest.skip(
            "RVN_PLE_TREE unset: apply patches/0047..0057 to a tree root with "
            "-p1 to check the graft stamp against the shipped rvn_mtp_count",
            allow_module_level=True,
        )
    root = Path(tree)
    if not root.is_dir():
        raise AssertionError(f"RVN_PLE_TREE={tree} is not a directory")
    return root


TREE = _tree_root()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graft_mtp = _load("rvn_graft_mtp_under_test", GRAFT_TOOL)
verify = _load("rvn_graft_verify_under_test", VERIFY_TOOL)


def _adapter():
    """The applied tree's adapter, or a skip when the tree predates patch 0047."""
    path = TREE / ADAPTER_REL
    if not path.is_file():
        pytest.skip(f"RVN_PLE_TREE={TREE} lacks {ADAPTER_REL} (patch 0047)")
    return _load("rvn_graft_tree_adapter", path)


# ------------------------------------------------------------------ fixtures


def _bf16(count, seed=0):
    return torch.arange(count, dtype=torch.float32).add(seed).to(torch.bfloat16)


def _f8e4m3(count, seed=0):
    """FP8 E4M3 by payload byte, so every stored encoding is a legal finite one."""
    return torch.tensor(
        [0x30 + (i + seed) % 8 for i in range(count)], dtype=torch.uint8
    ).view(torch.float8_e4m3fn)


def _u8(count, seed=0):
    return torch.tensor([(3 * i + seed) % 256 for i in range(count)], dtype=torch.uint8)


def _save(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous().clone() for name, tensor in tensors.items()},
        str(path),
    )


def _payload_sha(tensor):
    """sha256 over a tensor's raw contiguous payload bytes."""
    flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _header(path):
    """``(entries, data_start)`` of a safetensors file, header parsed by hand."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        meta = json.loads(handle.read(length))
    return {k: v for k, v in meta.items() if k != "__metadata__"}, 8 + length


def _read_payload(path, entry, data_start):
    begin, end = entry["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_start + begin)
        payload = handle.read(end - begin)
    assert len(payload) == end - begin, f"short payload in {path}"
    return payload


SOURCE_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
SOURCE_MTP = {
    # Deliberately split across two shards: the graft writes one shard per source
    # shard, while the real checkpoint keeps all 4,637 in a single one.
    SOURCE_SHARDS[0]: {
        "mtp.layers.0.mlp.gate_proj.weight": _bf16(12, 1).reshape(3, 4),
        "mtp.layers.0.self_attn.q_proj.weight": _f8e4m3(8, 2),
    },
    SOURCE_SHARDS[1]: {
        "mtp.fc_embedding.weight": _bf16(24, 4).reshape(4, 6),
        "mtp.pre_fc_norm_embedding.weight": _u8(6, 5),
        "mtp.fc_hidden.weight": torch.arange(8, dtype=torch.float32).mul(0.5),
    },
}
SOURCE_OTHER = {
    SOURCE_SHARDS[0]: {
        "model.language_model.embed_tokens.weight": _bf16(24, 9).reshape(4, 6),
    },
    SOURCE_SHARDS[1]: {
        "lm_head.weight": _bf16(24, 8).reshape(4, 6),
    },
}
GRAFTED_NAMES = sorted(name for tensors in SOURCE_MTP.values() for name in tensors)
GRAFT_SHARDS = tuple(
    f"rvn-mtp-graft-{index + 1:05d}-of-{len(SOURCE_SHARDS):05d}.safetensors"
    for index in range(len(SOURCE_SHARDS))
)

BASE_SHARD = "model-00001-of-00001.safetensors"
BASE_TENSORS = {
    "model.language_model.embed_tokens.weight": _bf16(24, 11).reshape(4, 6),
    "model.language_model.layers.0.mlp.gate_proj.weight": _bf16(12, 12).reshape(3, 4),
    "lm_head.weight": _bf16(24, 13).reshape(4, 6),
}
PLE_PART_NAME = "rvn_ple_parts/part-00001.safetensors"
PLE_PART_TENSORS = {
    "rvn_ple.packed.w1": _u8(64, 21),
    "rvn_ple.packed.s1": _f8e4m3(4, 22),
}


def _base_config(**overrides):
    config = {
        "architectures": [RVN_ARCH],
        "model_type": RVN_MODEL_TYPE,
        "hidden_size": 2560,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 0,
        # The shipped candidate really does carry this inert sub-object, and the
        # contract deliberately leaves it at 0.
        "mtp": {"num_hidden_layers": 0, "hybrid": True},
        "vocab_size": 248320,
        "ple_embedding_dtype": "nvfp4",
    }
    config.update(overrides)
    return config


def _write_index(root, weight_map, tensors_by_shard):
    total = 0
    for shard, tensors in tensors_by_shard.items():
        entries, _ = _header(root / shard)
        for name in tensors:
            assert name in entries
            total += entries[name]["data_offsets"][1] - entries[name]["data_offsets"][0]
    (root / INDEX_NAME).write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}) + "\n"
    )


def make_source(root, extra_mtp=None):
    """Fake MTP source: two shards with mtp.* plus ordinary tensors, and an index.

    ``extra_mtp`` adds tensors to the first shard (index included), which is how
    the multi-layer-draft-head refusal is provoked.
    """
    root = Path(root)
    extra_mtp = extra_mtp or {}
    weight_map = {}
    by_shard = {}
    for shard in SOURCE_SHARDS:
        tensors = dict(SOURCE_MTP[shard], **SOURCE_OTHER[shard])
        if shard == SOURCE_SHARDS[0]:
            tensors.update(extra_mtp)
        _save(root / shard, tensors)
        by_shard[shard] = tensors
        weight_map.update({name: shard for name in tensors})
    _write_index(root, weight_map, by_shard)
    (root / CONFIG_NAME).write_text(
        json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"]}) + "\n"
    )
    return root


def make_base(root, *, config=None, extra_index=None):
    """Fake ungrafted RVN candidate: shard, index, PLE manifest and nested parts."""
    root = Path(root)
    _save(root / BASE_SHARD, BASE_TENSORS)
    weight_map = {name: BASE_SHARD for name in BASE_TENSORS}
    weight_map.update(extra_index or {})
    _write_index(root, weight_map, {BASE_SHARD: BASE_TENSORS})
    (root / CONFIG_NAME).write_text(json.dumps(config or _base_config()) + "\n")
    _save(root / PLE_PART_NAME, PLE_PART_TENSORS)
    (root / "ple_storage.json").write_text(
        json.dumps({"format_version": 1, "parts": [{"file": PLE_PART_NAME}]}) + "\n"
    )
    (root / "tokenizer.json").write_text('{"tokenizer": "test"}\n')
    # Converter scratch: dotfiles must never be carried into the graft.
    (root / ".rvn_convert_state.json").write_text('{"key": "scratch"}\n')
    return root


@pytest.fixture
def tree(tmp_path):
    return (
        make_source(tmp_path / "source"),
        make_base(tmp_path / "base"),
        tmp_path / "out",
    )


def graft_ok(source, base, out, **kwargs):
    return graft_mtp.graft(source=str(source), base=str(base), out=str(out), **kwargs)


def verify_graft(source, base, out):
    return verify.run_graft_verification(str(source), str(base), str(out))


def statuses(report):
    return {check["name"]: check["status"] for check in report["checks"]}


def first_failure(report):
    failed = [check for check in report["checks"] if check["status"] == "fail"]
    assert failed, report
    return failed[0]


# ---------------------------------------------------------------------- tests


def test_graft_writes_the_contract_outputs(tree):
    source, base, out = tree
    graft_ok(source, base, out)

    # Every base file arrives, nested part included, as a hardlink of the base.
    for rel in (BASE_SHARD, "ple_storage.json", PLE_PART_NAME, "tokenizer.json"):
        src_stat, out_stat = (base / rel).stat(), (out / rel).stat()
        assert (src_stat.st_dev, src_stat.st_ino) == (out_stat.st_dev, out_stat.st_ino), rel
        assert out_stat.st_nlink > 1, rel
    assert not (out / ".rvn_convert_state.json").exists()

    manifest = json.loads((out / GRAFT_MANIFEST_NAME).read_text())
    assert set(manifest) == {"source", "tensors"}
    assert manifest["source"] == str(Path(source).resolve())
    assert sorted(manifest["tensors"]) == GRAFTED_NAMES
    for name, entry in manifest["tensors"].items():
        assert set(entry) == {"file", "sha256", "dtype", "shape"}
        assert entry["file"] in SOURCE_SHARDS
        assert entry["sha256"] == _payload_sha(SOURCE_MTP[entry["file"]][name])
        assert entry["shape"] == list(SOURCE_MTP[entry["file"]][name].shape)

    config = json.loads((out / CONFIG_NAME).read_text())
    assert config["mtp_num_hidden_layers"] == 1
    assert config[GRAFT_STAMP_KEY] == {
        "source": str(Path(source).resolve()),
        "encoder_version": ENCODER_VERSION,
        "count": 1,
    }
    # The inert nested sub-object is left exactly as the base declared it.
    assert config["mtp"] == {"num_hidden_layers": 0, "hybrid": True}

    base_index = json.loads((base / INDEX_NAME).read_text())
    index = json.loads((out / INDEX_NAME).read_text())
    assert set(index["weight_map"]) == set(base_index["weight_map"]) | set(GRAFTED_NAMES)
    for name, shard in base_index["weight_map"].items():
        assert index["weight_map"][name] == shard
    grafted_shards = sorted({index["weight_map"][name] for name in GRAFTED_NAMES})
    assert grafted_shards == list(GRAFT_SHARDS)
    payload_bytes = sum(
        entry["data_offsets"][1] - entry["data_offsets"][0]
        for shard in grafted_shards
        for entry in _header(out / shard)[0].values()
    )
    assert index["metadata"]["total_size"] == (
        base_index["metadata"]["total_size"] + payload_bytes
    )
    for shard in grafted_shards:
        # The graft's own shard must be private: it is the only file a repair path
        # may rewrite, and a base hardlink there would corrupt the base.
        assert (out / shard).stat().st_nlink == 1


def test_graft_payloads_are_the_source_bytes(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    index = json.loads((out / INDEX_NAME).read_text())["weight_map"]
    for shard in SOURCE_SHARDS:
        src_entries, src_start = _header(source / shard)
        for name in SOURCE_MTP[shard]:
            dst_path = out / index[name]
            dst_entries, dst_start = _header(dst_path)
            src_entry, dst_entry = src_entries[name], dst_entries[name]
            assert dst_entry["dtype"] == src_entry["dtype"]
            assert dst_entry["shape"] == src_entry["shape"]
            assert _read_payload(dst_path, dst_entry, dst_start) == _read_payload(
                source / shard, src_entry, src_start
            )


def test_graft_never_carries_non_mtp_names(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    manifest = json.loads((out / GRAFT_MANIFEST_NAME).read_text())
    for shard in SOURCE_SHARDS:
        for name in SOURCE_OTHER[shard]:
            assert name not in manifest["tensors"]
    index = json.loads((out / INDEX_NAME).read_text())["weight_map"]
    grafted = {name for name in index if name.startswith("mtp.")}
    assert grafted == set(GRAFTED_NAMES)
    # The base shard holds no draft tensors: the graft added a shard, not weights.
    assert all(name in BASE_TENSORS for name in index if index[name] == BASE_SHARD)


def test_graft_is_byte_identical_on_rerun(tree, tmp_path):
    source, base, out = tree
    graft_ok(source, base, out)
    second = tmp_path / "out2"
    graft_ok(source, base, second)

    def digest(root):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(root).iterdir())
            if path.is_file()
        }

    first = digest(out)
    assert set(first) == {
        *GRAFT_SHARDS, BASE_SHARD, "ple_storage.json", "tokenizer.json",
        CONFIG_NAME, INDEX_NAME, GRAFT_MANIFEST_NAME,
    }
    assert first == digest(second)


def test_graft_verify_passes(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    report = verify_graft(source, base, out)
    assert report["ok"], report
    assert tuple(check["name"] for check in report["checks"]) == GRAFT_CHECK_ORDER
    assert report["tensors"] == len(GRAFTED_NAMES)


def test_graft_verify_detects_one_corrupted_byte(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    victim = "mtp.layers.0.mlp.gate_proj.weight"
    shard = GRAFT_SHARDS[0]
    entries, data_start = _header(out / shard)
    offset = entries[victim]["data_offsets"][0]
    with open(out / shard, "r+b") as handle:  # the graft's own file, never a base link
        handle.seek(data_start + offset)
        byte = handle.read(1)
        handle.seek(data_start + offset)
        handle.write(bytes([byte[0] ^ 0x01]))

    report = verify_graft(source, base, out)
    assert not report["ok"]
    assert statuses(report)["graft_payload_identity"] == "fail"
    assert victim in first_failure(report)["reason"]


def test_graft_verify_detects_a_requantized_payload(tree):
    """Same length, different bytes: exactly what a sha-per-tensor gate is for."""
    source, base, out = tree
    graft_ok(source, base, out)
    victim = "mtp.fc_hidden.weight"
    shard = index_shard(out, victim)
    entries, data_start = _header(out / shard)
    begin, end = entries[victim]["data_offsets"]
    with open(out / shard, "r+b") as handle:
        handle.seek(data_start + begin)
        payload = handle.read(end - begin)
        handle.seek(data_start + begin)
        handle.write(b"\x00" * len(payload))

    report = verify_graft(source, base, out)
    assert statuses(report)["graft_payload_identity"] == "fail"
    assert victim in first_failure(report)["reason"]


def index_shard(out, name):
    return json.loads((out / INDEX_NAME).read_text())["weight_map"][name]


def _drop_tensor(source, base, out):
    path = out / GRAFT_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["tensors"].pop("mtp.fc_hidden.weight")
    path.write_text(json.dumps(manifest))


def _stale_encoder(source, base, out):
    path = out / CONFIG_NAME
    config = json.loads(path.read_text())
    config[GRAFT_STAMP_KEY]["encoder_version"] = "rvn-mtp-graft-r0"
    path.write_text(json.dumps(config))


def _wrong_source(source, base, out):
    path = out / GRAFT_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["source"] = "/models/some-other-checkpoint"
    path.write_text(json.dumps(manifest))


def _forge_shard(source, base, out):
    """Point a grafted tensor at a base shard (a hardlinked target shard)."""
    path = out / INDEX_NAME
    index = json.loads(path.read_text())
    index["weight_map"]["mtp.fc_hidden.weight"] = BASE_SHARD
    path.write_text(json.dumps(index))


def _drop_base_file(source, base, out):
    (out / PLE_PART_NAME).unlink()


def _rewrite_base_file(source, base, out):
    path = out / "ple_storage.json"
    path.unlink()  # break the hardlink first, or the base candidate itself changes
    path.write_text('{"format_version": 999}\n')


def _extra_index_entry(source, base, out):
    path = out / INDEX_NAME
    index = json.loads(path.read_text())
    index["weight_map"]["model.leftover.weight"] = BASE_SHARD
    path.write_text(json.dumps(index))


def _zero_mtp_depth(source, base, out):
    path = out / CONFIG_NAME
    config = json.loads(path.read_text())
    config["mtp_num_hidden_layers"] = 0
    path.write_text(json.dumps(config))


def _base_already_has_mtp(source, base, out):
    """The overlap rule: a base carrying its own draft head is refused, not merged."""
    for root in (base, out):
        path = root / INDEX_NAME
        index = json.loads(path.read_text())
        index["weight_map"]["mtp.layers.0.preexisting.weight"] = BASE_SHARD
        path.write_text(json.dumps(index))


TAMPER_CASES = [
    pytest.param(_drop_tensor, "mtp.fc_hidden.weight", id="dropped-tensor"),
    pytest.param(_stale_encoder, "rvn-mtp-graft-r0", id="stale-encoder-version"),
    pytest.param(_wrong_source, "some-other-checkpoint", id="wrong-source"),
    pytest.param(_forge_shard, "base shard", id="grafted-tensor-in-base-shard"),
    pytest.param(_drop_base_file, "part-00001.safetensors", id="missing-base-file"),
    pytest.param(_rewrite_base_file, "ple_storage.json", id="rewritten-base-file"),
    pytest.param(_extra_index_entry, "model.leftover.weight", id="injected-index-entry"),
    pytest.param(_zero_mtp_depth, "0 MTP layer", id="stamp-depth-zero"),
    pytest.param(_base_already_has_mtp, "already carries", id="base-carries-mtp"),
]


@pytest.mark.parametrize("tamper,keyword", TAMPER_CASES)
def test_graft_verify_refuses_every_tamper_shape(tree, tamper, keyword):
    source, base, out = tree
    graft_ok(source, base, out)
    tamper(source, base, out)
    report = verify_graft(source, base, out)
    assert not report["ok"], report
    assert keyword in first_failure(report)["reason"], report


def test_graft_verify_fixture_is_clean_before_tampering(tree):
    """Guards the tamper suite: every failure above must come from the tamper."""
    source, base, out = tree
    graft_ok(source, base, out)
    assert verify_graft(source, base, out)["ok"]


def test_graft_verify_refuses_an_undeclared_extra_shard(tree):
    """An unreferenced shard still reaches the loader: weight iterators walk files."""
    source, base, out = tree
    graft_ok(source, base, out)
    extra = "rvn-mtp-graft-00009-of-00009.safetensors"
    _save(out / extra, {"mtp.sneaky.weight": _bf16(4, 40)})
    report = verify_graft(source, base, out)
    assert not report["ok"], report
    assert extra in first_failure(report)["reason"]


def test_graft_verify_refuses_an_undeclared_tensor_in_a_grafted_shard(tree):
    """Same-length re-save with one smuggled tensor: the manifest cannot vouch for it."""
    source, base, out = tree
    graft_ok(source, base, out)
    shard = out / GRAFT_SHARDS[0]
    tensors = load_file(str(shard))
    tensors["mtp.sneaky.weight"] = _bf16(4, 41)
    _save(shard, tensors)
    report = verify_graft(source, base, out)
    assert not report["ok"], report
    assert statuses(report)["graft_index"] == "fail"
    assert "mtp.sneaky.weight" in first_failure(report)["reason"]


def test_graft_refuses_a_base_that_already_has_a_draft_head(tmp_path):
    base = make_base(
        tmp_path / "base", extra_index={"mtp.layers.0.preexisting.weight": BASE_SHARD}
    )
    source = make_source(tmp_path / "source")
    with pytest.raises(graft_mtp.GraftError, match="already carries"):
        graft_ok(source, base, tmp_path / "out")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mtp_num_hidden_layers": 1}, "already declares 1 MTP layer"),
        ({"num_nextn_predict_layers": 2}, "already declares 2 MTP layer"),
        ({GRAFT_STAMP_KEY: {"count": 1}}, "already carries a rvn_mtp_graft stamp"),
        ({"architectures": ["Qwen4ExpForConditionalGeneration"]}, "not the RVN text"),
        ({"model_type": "qwen4_exp"}, "not the RVN text"),
    ],
)
def test_graft_refuses_bases_that_are_not_the_ungrafted_candidate(
    tmp_path, overrides, match
):
    base = make_base(tmp_path / "base", config=_base_config(**overrides))
    source = make_source(tmp_path / "source")
    with pytest.raises(graft_mtp.GraftError, match=match):
        graft_ok(source, base, tmp_path / "out")


def test_graft_refuses_a_multi_layer_draft_head(tmp_path):
    """count: 1 is a claim about the source head, so a second layer is refused."""
    source = make_source(
        tmp_path / "source",
        extra_mtp={"mtp.layers.1.mlp.gate_proj.weight": _bf16(4, 30)},
    )
    with pytest.raises(graft_mtp.GraftError, match="not the one-layer head"):
        graft_ok(source, make_base(tmp_path / "base"), tmp_path / "out")


def test_graft_refuses_a_source_shard_the_index_does_not_map(tmp_path):
    """An unmapped mtp tensor in an opened shard would be silently dropped."""
    source = make_source(tmp_path / "source")
    shard = SOURCE_SHARDS[0]
    tensors = dict(SOURCE_MTP[shard], **SOURCE_OTHER[shard])
    tensors["mtp.layers.0.unmapped.weight"] = _bf16(4, 31)
    _save(source / shard, tensors)
    with pytest.raises(graft_mtp.GraftError, match="does not map"):
        graft_ok(source, make_base(tmp_path / "base"), tmp_path / "out")


def test_graft_refuses_a_source_without_any_draft_head(tmp_path):
    source = make_source(tmp_path / "source")
    for shard in SOURCE_SHARDS:
        tensors = dict(SOURCE_OTHER[shard])
        _save(source / shard, tensors)
    _write_index(source, {n: s for s in SOURCE_SHARDS for n in SOURCE_OTHER[s]},
                 {s: SOURCE_OTHER[s] for s in SOURCE_SHARDS})
    with pytest.raises(graft_mtp.GraftError, match="no mtp.* tensors"):
        graft_ok(source, make_base(tmp_path / "base"), tmp_path / "out")


def test_graft_refuses_to_write_into_source_or_base(tree):
    source, base, _ = tree
    with pytest.raises(graft_mtp.GraftError, match="separate directory"):
        graft_ok(source, base, base)
    with pytest.raises(graft_mtp.GraftError, match="separate directory"):
        graft_ok(source, base, source / "graft")


def test_graft_refuses_an_unrelated_non_empty_out(tmp_path):
    source = make_source(tmp_path / "source")
    base = make_base(tmp_path / "base")
    out = tmp_path / "someone-elses-checkpoint"
    out.mkdir()
    (out / "config.json").write_text('{"architectures": ["Other"]}\n')
    with pytest.raises(graft_mtp.GraftError, match="without mtp_graft.json"):
        graft_ok(source, base, out)


def test_graft_repeats_over_a_previous_graft(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    (out / GRAFT_SHARDS[0]).write_bytes(b"stale")
    graft_ok(source, base, out)  # re-graft, not a refusal
    assert verify_graft(source, base, out)["ok"]


def test_link_denial_fails_closed_naming_the_file(tree, monkeypatch):
    source, base, out = tree
    real_link = os.link

    def deny(src, dst, *args, **kwargs):
        if Path(src).name == "ple_storage.json":
            raise OSError(1, "Operation not permitted")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", deny)
    with pytest.raises(graft_mtp.GraftError, match="cannot hardlink.*ple_storage.json"):
        graft_ok(source, base, out)


def test_cross_device_out_copies_and_says_so(tree, monkeypatch):
    source, base, out = tree
    monkeypatch.setattr(
        os, "link",
        lambda *a, **k: (_ for _ in ()).throw(OSError(18, "Invalid cross-device link")),
    )
    summary = graft_ok(source, base, out)
    assert summary["copied_bytes"] > 0
    for rel in (BASE_SHARD, PLE_PART_NAME):
        assert (out / rel).is_file()
        assert (out / rel).stat().st_nlink == 1  # copied, so not shared with the base
    assert verify_graft(source, base, out)["ok"]


def test_no_copy_refuses_a_cross_device_out(tree, monkeypatch):
    source, base, out = tree
    monkeypatch.setattr(
        os, "link",
        lambda *a, **k: (_ for _ in ()).throw(OSError(18, "Invalid cross-device link")),
    )
    with pytest.raises(graft_mtp.GraftError, match="would copy"):
        graft_ok(source, base, out, no_copy=True)


def test_stamped_config_satisfies_the_tree_mtp_count(tree):
    """The stamp is the loader rule's precondition; checked against shipped code."""
    adapter = _adapter()
    source, base, out = tree
    graft_ok(source, base, out)
    assert adapter.rvn_mtp_count(json.loads((base / CONFIG_NAME).read_text())) == 0
    assert adapter.rvn_mtp_count(json.loads((out / CONFIG_NAME).read_text())) == 1


def test_graft_stamp_unlocks_the_trees_loader_rule(tree):
    """Writer and loader must agree literally, or the graft never loads.

    Skips on a tree without patch 0058, because the exemption predicate does not
    exist there yet and the graft would be refused at weight load.
    """
    adapter = _adapter()
    active = getattr(adapter, "rvn_mtp_graft_active", None)
    if active is None:
        pytest.skip(
            f"RVN_PLE_TREE={TREE} predates "
            "patches/0058-rvn-mtp-graft-loader.patch: no loader exemption to agree with"
        )
    source, base, out = tree
    graft_ok(source, base, out)
    grafted = json.loads((out / CONFIG_NAME).read_text())
    assert active(grafted) is True
    assert active(json.loads((base / CONFIG_NAME).read_text())) is False
    # The entry class's own invariant tolerates exactly this config.
    adapter.assert_rvn_text_config(grafted)
    for mutation, why in (
        (("encoder_version", "rvn-mtp-graft-r0"), "foreign encoder version"),
        (("encoder_version", None), "missing encoder version"),
        (("count", 2), "count other than 1"),
    ):
        config = json.loads(json.dumps(grafted))
        key, value = mutation
        if value is None:
            del config[GRAFT_STAMP_KEY][key]
        else:
            config[GRAFT_STAMP_KEY][key] = value
        assert active(config) is False, why
    # A grafted head declared without the stamp stays refused.
    unstamped = json.loads(json.dumps(grafted))
    del unstamped[GRAFT_STAMP_KEY]
    assert active(unstamped) is False


def test_graft_cli_writes_the_out_dir(tree):
    source, base, out = tree
    proc = subprocess.run(
        [sys.executable, str(GRAFT_TOOL), "--source", str(source), "--base", str(base),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "graft OK" in proc.stdout
    assert f"wrote {GRAFT_MANIFEST_NAME}" in proc.stdout


def test_graft_cli_failure_is_a_message_not_a_traceback(tree):
    source, base, _ = tree
    proc = subprocess.run(
        [sys.executable, str(GRAFT_TOOL), "--source", str(source), "--base", str(base),
         "--out", str(base)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "graft FAILED" in proc.stderr


def test_graft_verify_cli_reports_and_exits(tree):
    source, base, out = tree
    graft_ok(source, base, out)
    proc = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--graft", "--source", str(source),
         "--base", str(base), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "verification PASSED: 7/7 checks" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_graft_verify_cli_missing_arguments_is_a_clean_usage_error(tree):
    """Suite-wide rule: bad CLI usage exits 2 with argparse's message, no traceback."""
    source, base, out = tree
    proc = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--graft", "--source", str(source)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "--graft needs" in proc.stderr
    proc = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--src-dir", str(source)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    # --graft must not silently reuse the PLE-mode directories.
    proc = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--graft", "--source", str(source),
         "--base", str(base), "--out", str(out), "--src-dir", str(source),
         "--dst-dir", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "not --src-dir/--dst-dir" in proc.stderr


# ---------------------------------------------------------- real-path, read-only


def _real_weight_map(root):
    try:
        return json.loads((root / INDEX_NAME).read_text())["weight_map"]
    except (OSError, ValueError, KeyError):
        return None


def test_real_mtp_source_declares_the_frozen_draft_head():
    """Index JSONs only: this must not become a 103 GB payload hash by default."""
    if not (REAL_SOURCE / INDEX_NAME).is_file():
        pytest.skip(f"{REAL_SOURCE} is not readable from here")
    weight_map = _real_weight_map(REAL_SOURCE)
    if weight_map is None:
        pytest.skip(f"{REAL_SOURCE}/{INDEX_NAME} unreadable")
    assert sum(1 for name in weight_map if name.startswith("mtp.")) == REAL_MTP_TENSORS
    base_map = _real_weight_map(REAL_BASE)
    if base_map is not None:
        assert not any("mtp" in name for name in base_map)


def test_real_graft_manifest_if_published():
    if not (REAL_OUT / GRAFT_MANIFEST_NAME).is_file():
        pytest.skip("the real graft has not been published yet")
    manifest = json.loads((REAL_OUT / GRAFT_MANIFEST_NAME).read_text())
    assert len(manifest["tensors"]) == REAL_MTP_TENSORS
    config = json.loads((REAL_OUT / CONFIG_NAME).read_text())
    assert config[GRAFT_STAMP_KEY]["encoder_version"] == ENCODER_VERSION
    assert config[GRAFT_STAMP_KEY]["count"] == 1
    assert config["mtp_num_hidden_layers"] == 1


@pytest.mark.skipif(
    not os.environ.get("RVN_PLE_GRAFT_REAL"),
    reason="the full real byte-identity pass re-hashes 1.5 GiB twice; opt in with "
           "RVN_PLE_GRAFT_REAL=1",
)
def test_real_graft_byte_identity():
    if not (REAL_OUT / GRAFT_MANIFEST_NAME).is_file():
        pytest.skip("the real graft has not been published yet")
    report = verify_graft(REAL_SOURCE, REAL_BASE, REAL_OUT)
    assert report["ok"], report

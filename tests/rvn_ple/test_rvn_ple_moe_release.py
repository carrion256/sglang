"""RVN-only NVFP4-Marlin MoE repack release regression (patch 0052).

Why this test exists
-------------------
The RVN launch pins ``--moe-runner-backend=marlin`` (patch 0048 makes the
W4A16_NVFP4 checkpoint honour it), which routes every MoE layer through
``prepare_moe_nvfp4_layer_for_marlin``. The LIL launch runs the same checkpoint
family with ``--moe-runner-backend=flashinfer_cutlass`` and never enters that
function, so the marlin repack's memory behaviour is RVN-only surface.

Measured on a synthetic multi-layer NVFP4 MoE (real ``create_weights``, real
``process_weights_after_loading``, backend pinned to marlin), the preimage
branch:

  * held ``w13_blockscale_swizzled`` / ``w2_blockscale_swizzled`` -- loader-format
    scale copies that create_weights allocated and the marlin apply path never
    reads -- live for the process lifetime (2 x scale bytes per layer: ~7 GiB
    across the 48-layer RVN checkpoint); and
  * peaked ~2 full weight copies above the pre-layer baseline, because
    ``_repack_weight``/``_permute_scales`` built a per-expert list and then
    ``torch.stack``ed it while the loader-format original was still referenced
    by the enclosing function's locals.

Patch 0052 keeps the bytes identical (same per-expert kernel output, same
contiguous stack layout) but releases the dead placeholders up front, fills a
preallocated stack buffer expert-by-expert, and drops each original as soon as
its replacement is bound.

Run: RVN_PLE_TREE=/path/to/stacked-tree python3 -m pytest tests/rvn_ple/test_rvn_ple_moe_release.py -q
Skips cleanly without CUDA so the headless battery stays green.
"""

import importlib.util
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("marlin repack peak regression requires CUDA", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
TREE = Path(os.environ.get("RVN_PLE_TREE", str(REPO)))
TREE_ROOT = TREE / "python"

HIDDEN = 2560
INTERMEDIATE = 640
NUM_EXPERTS = 16
LAYERS = 4
GIB = 2**30


def _tree_file(rel):
    candidate = TREE_ROOT / rel
    return candidate if candidate.is_file() else None


def _load_from_tree(rel, name):
    """Load a stacked-tree module with its dependencies from installed sglang."""
    path = _tree_file(rel)
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_function():
    """The stacked tree's prepare_moe_nvfp4_layer_for_marlin (patch 0052)."""
    module = _load_from_tree(
        "sglang/srt/layers/quantization/marlin_utils_fp4.py",
        "rvn_ple_moe_release_marlin_fp4",
    )
    if module is None:
        pytest.skip(f"stacked tree missing under {TREE} (set RVN_PLE_TREE)")
    return module.prepare_moe_nvfp4_layer_for_marlin


def _make_layer(backend, seed):
    """One real NVFP4 MoE layer, built by the real create_weights path."""
    from sglang.srt.layers.moe import MoeRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptNvFp4FusedMoEMethod,
    )

    import inspect

    kwargs = dict(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
    )
    if "quant_format" in inspect.signature(ModelOptFp4Config.__init__).parameters:
        kwargs["quant_format"] = "W4A16_NVFP4"  # patch 0048
    config = ModelOptFp4Config(**kwargs)

    method = object.__new__(ModelOptNvFp4FusedMoEMethod)
    method.quant_config = config
    method.enable_flashinfer_trtllm_moe = False
    method._cache_permute_indices = {}
    method._moe_runner_backend = (
        MoeRunnerBackend.MARLIN
        if backend == "marlin"
        else MoeRunnerBackend.FLASHINFER_CUTLASS
    )

    layer = torch.nn.Module()
    layer.num_local_experts = NUM_EXPERTS
    layer.num_experts = NUM_EXPERTS
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True, activation="silu", num_experts=NUM_EXPERTS, top_k=10
    )
    # The loader builds models under `with target_device:`; parameters must be
    # allocated the same way or the peak measures a different allocation mix.
    with torch.device("cuda"):
        method.create_weights(
            layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
    torch.manual_seed(seed)
    with torch.no_grad():
        # Filled in place: a materialising fill would perturb the peak measured.
        for name, high in (
            ("w13_weight", 256),
            ("w2_weight", 252),
            ("w13_weight_scale", 100),
            ("w2_weight_scale", 90),
        ):
            layer._parameters[name].data.view(torch.uint8).random_(0, high)
        layer.w13_weight_scale_2.data.fill_(3e-4)
        layer.w2_weight_scale_2.data.fill_(3e-4)
        layer.w13_input_scale.data.fill_(1.0)
        layer.w2_input_scale.data.fill_(1.0)
    return layer, method


def _load_and_postprocess(backend, prepare=None):
    """Real per-module postprocess loop, as load_weights_and_postprocess runs it."""
    from sglang.srt.layers.moe import MoeRunnerBackend
    from sglang.srt.layers.quantization import modelopt_quant as modelopt

    original_backend = modelopt.get_moe_runner_backend
    modelopt.get_moe_runner_backend = lambda: (
        MoeRunnerBackend.MARLIN
        if backend == "marlin"
        else MoeRunnerBackend.FLASHINFER_CUTLASS
    )
    original = modelopt.prepare_moe_nvfp4_layer_for_marlin
    if prepare is not None:
        modelopt.prepare_moe_nvfp4_layer_for_marlin = prepare
    try:
        built = [_make_layer(backend, 1000 + i) for i in range(LAYERS)]
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()

        # Bytes create_weights put into the loader-format swizzle placeholders.
        # They are sampled here, before any postprocess, because the point of
        # the residual assertion below is that postprocess must hand *these*
        # bytes back, not merely avoid adding new ones.
        placeholder_bytes = sum(
            getattr(layer, name).numel() * getattr(layer, name).element_size()
            for layer, _ in built
            for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled")
            if getattr(layer, name, None) is not None
        )
        loader_names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
        originals = [
            {name: weakref.ref(layer._parameters[name].data) for name in loader_names}
            for layer, _ in built
        ]
        peaks, residuals, dead = [], [], []
        for index, (layer, method) in enumerate(built):
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            method.process_weights_after_loading(layer)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - before)
            residuals.append(torch.cuda.memory_allocated() - base)
            dead.append(
                {name: ref() is None for name, ref in originals[index].items()}
            )
        return base, peaks, residuals, dead, built, placeholder_bytes
    finally:
        modelopt.prepare_moe_nvfp4_layer_for_marlin = original
        modelopt.get_moe_runner_backend = original_backend


def test_marlin_repack_peak_and_release_loader_format_storage():
    """Peak stays at one fresh weight copy; loader-format storage is released."""
    prepare = _prepare_function()
    base, peaks, residuals, _, built, _ = _load_and_postprocess("marlin", prepare)

    # Largest single loader-format weight (w13): the one fresh copy the repack
    # legitimately needs live at a time. One expert's payload is the working
    # set it must stay within.
    one_expert = (2 * INTERMEDIATE) * (HIDDEN // 2)
    largest_weight = NUM_EXPERTS * one_expert
    bound = largest_weight + 2 * one_expert

    for index, peak in enumerate(peaks):
        assert peak <= bound, (
            f"layer {index}: marlin repack peaked {peak / GIB:.4f} GiB above the "
            f"pre-layer baseline, above the one-fresh-copy bound "
            f"{bound / GIB:.4f} GiB (w13={largest_weight / GIB:.4f} GiB, "
            f"expert={one_expert / GIB:.4f} GiB)"
        )
    # Structural check that the preallocated-stack rewrite still yields the
    # exact marlin layout: int32 packed weights with the marlin repack shape.
    first = built[0][0]
    w13 = first._parameters["w13_weight"].data
    w2 = first._parameters["w2_weight"].data
    assert w13.dtype == torch.int32 and w2.dtype == torch.int32
    assert tuple(w13.shape) == (NUM_EXPERTS, HIDDEN // 16, 2 * INTERMEDIATE * 16 // 8)
    assert tuple(w2.shape) == (NUM_EXPERTS, INTERMEDIATE // 16, HIDDEN * 16 // 8)

    # The loader-format swizzle placeholders are dead on the marlin path: the
    # apply path reads w*_weight_scale, never the swizzled copies.
    for layer, _ in built:
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            assert getattr(layer, name, None) is None, (
                f"{name} still holds {(getattr(layer, name).numel() * getattr(layer, name).element_size()) / 2**20:.1f} MiB"
                " of loader-format scales the marlin path never reads"
            )


def test_marlin_repack_hands_back_the_dead_swizzle_placeholders():
    """Postprocess must free the placeholder bytes, not just avoid growing."""
    base, _, residuals, _, _, placeholder_bytes = _load_and_postprocess(
        "marlin", _prepare_function()
    )
    assert placeholder_bytes > 0, "fixture allocated no swizzle placeholders"
    # The repacked copies are byte-for-byte the size of the loader-format
    # originals they replace, so the only reason live bytes should fall below
    # the baseline is the dead w13/w2_blockscale_swizzled pair per layer. Assert
    # against those bytes: a release that only cancels bookkeeping noise would
    # pass a bare "<= 0" threshold without giving the ~2 x scale bytes back.
    assert residuals[-1] <= -placeholder_bytes // 2, (
        f"marlin postprocess returned {residuals[-1] / GIB:+.4f} GiB, expected at "
        f"least {placeholder_bytes / 2 / GIB:.4f} GiB of the "
        f"{placeholder_bytes / GIB:.4f} GiB of swizzle placeholders to be gone "
        f"(base {base / GIB:.3f} GiB)"
    )


def test_marlin_repack_releases_every_loader_format_original():
    """No original tensor outlives its layer's repack."""
    _, _, _, dead, _, _ = _load_and_postprocess("marlin", _prepare_function())
    for index, names in enumerate(dead):
        still_live = [name for name, alive in names.items() if not alive]
        assert not still_live, f"layer {index}: originals still live: {still_live}"


def test_marlin_repack_is_unreachable_for_the_lil_cutlass_backend():
    """0052 can only affect the marlin path: cutlass never reaches it.

    The bare test module is not a full FusedMoE, so the cutlass branch raises
    once it registers its quant config; what matters -- and what keeps LIL
    byte-identical -- is that the marlin repack it would otherwise share is
    never entered for that backend.
    """
    from sglang.srt.layers.quantization import modelopt_quant as modelopt

    prepare = _prepare_function()
    calls = []

    def spy(layer):
        calls.append(layer)
        return prepare(layer)

    original = modelopt.prepare_moe_nvfp4_layer_for_marlin
    modelopt.prepare_moe_nvfp4_layer_for_marlin = spy
    try:
        try:
            _load_and_postprocess("cutlass")
            raise AssertionError("cutlass postprocess unexpectedly succeeded")
        except AttributeError as exc:
            # The stub module only fails once the cutlass branch runs, i.e.
            # *after* the marlin gate was passed without taking it. Binding the
            # message keeps an unrelated AttributeError (say, from building the
            # layer) from leaving this test vacuously green.
            assert "dispatcher" in str(exc), (
                f"cutlass branch failed before the marlin gate: {exc}"
            )
        assert not calls, "marlin repack ran for the flashinfer_cutlass backend"
        _load_and_postprocess("marlin")
    finally:
        modelopt.prepare_moe_nvfp4_layer_for_marlin = original
    assert len(calls) == LAYERS, f"marlin repack ran {len(calls)}x, expected {LAYERS}"


# --- patch 0053: the placeholders are never allocated under the marlin pin ---

RVN_EXPERTS = 512  # checkpoint config.num_experts
RVN_LAYERS = 48  # checkpoint config.num_hidden_layers
RVN_HIDDEN = 2560  # checkpoint config.hidden_size
RVN_INTERMEDIATE = 640  # checkpoint config.moe_intermediate_size
RVN_GROUP = 16  # hf_quant_config.json group_size
NON_EXPERT_GIB = 9.154  # sum of non-expert, non-PLE tensors in the 98 shards
CONTEXT_GIB = 0.9  # torch context + NCCL already resident at "Load weight begin"
BUDGET_GIB = 94.0  # 95.01 GiB device minus the margin the launch keeps


def _tree_modelopt():
    module = _load_from_tree(
        "sglang/srt/layers/quantization/modelopt_quant.py",
        "rvn_ple_moe_release_modelopt",
    )
    if module is None:
        pytest.skip(f"stacked tree missing under {TREE} (set RVN_PLE_TREE)")
    return module


def _build_one(tree_module, backend):
    """One layer built by the stacked tree's create_weights; returns (layer, bytes)."""
    from sglang.srt.layers.moe import MoeRunnerBackend
    import inspect

    kwargs = dict(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
    )
    if "quant_format" in inspect.signature(tree_module.ModelOptFp4Config.__init__).parameters:
        kwargs["quant_format"] = "W4A16_NVFP4"
    method = object.__new__(tree_module.ModelOptNvFp4FusedMoEMethod)
    method.quant_config = tree_module.ModelOptFp4Config(**kwargs)
    method.enable_flashinfer_trtllm_moe = False
    method._cache_permute_indices = {}
    method._moe_runner_backend = (
        MoeRunnerBackend.MARLIN
        if backend == "marlin"
        else MoeRunnerBackend.FLASHINFER_CUTLASS
    )
    layer = torch.nn.Module()
    layer.num_local_experts = NUM_EXPERTS
    layer.num_experts = NUM_EXPERTS
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True, activation="silu", num_experts=NUM_EXPERTS, top_k=10
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.device("cuda"):
        method.create_weights(
            layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
    torch.cuda.synchronize()
    return layer, torch.cuda.memory_allocated() - before


def _placeholder_bytes(layer):
    return sum(
        getattr(layer, name).numel() * getattr(layer, name).element_size()
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled")
        if getattr(layer, name, None) is not None
    )


def test_marlin_create_weights_never_allocates_the_placeholders():
    """0053: under the marlin pin the dead placeholders are never allocated."""
    tree_module = _tree_modelopt()
    marlin_layer, marlin_bytes = _build_one(tree_module, "marlin")
    cutlass_layer, cutlass_bytes = _build_one(tree_module, "cutlass")

    # LIL / cutlass unchanged: it still gets both loader-format scale copies.
    cutlass_placeholders = _placeholder_bytes(cutlass_layer)
    assert cutlass_placeholders > 0, "cutlass backend lost its swizzle placeholders"
    # 2 x scale bytes per layer for this fixture (w13 fp8 scales + w2 fp8 scales).
    expected = NUM_EXPERTS * (
        2 * INTERMEDIATE * (HIDDEN // RVN_GROUP) + HIDDEN * (INTERMEDIATE // RVN_GROUP)
    )
    assert cutlass_placeholders == expected, (
        f"cutlass placeholders are {cutlass_placeholders} bytes, expected {expected}"
    )

    assert _placeholder_bytes(marlin_layer) == 0, (
        f"marlin create_weights still allocates {_placeholder_bytes(marlin_layer)} "
        "bytes of swizzle placeholders the marlin path never reads"
    )
    assert marlin_bytes == cutlass_bytes - cutlass_placeholders, (
        f"marlin build allocated {marlin_bytes} bytes, expected the cutlass build "
        f"minus the {cutlass_placeholders} placeholder bytes ({cutlass_bytes - cutlass_placeholders})"
    )


def test_rvn_48_layer_marlin_projection_fits_the_device_budget():
    """Project the fixture's measured unit costs onto the real 48-layer model.

    Bytes are exact config arithmetic for the RVN dimensions; what the fixture
    licenses is the *shape* of the repack (one fresh weight copy live at a
    time, nothing left behind), which is asserted here as a measured ratio.
    """
    base, peaks, residuals, _, _, _ = _load_and_postprocess(
        "marlin", _prepare_function()
    )
    one_expert = (2 * INTERMEDIATE) * (HIDDEN // 2)
    fixture_w13 = NUM_EXPERTS * one_expert
    # The repack may not need materially more than its largest fresh copy.
    assert max(peaks) <= 1.15 * (fixture_w13 + 2 * one_expert), (
        f"fixture repack peaked {max(peaks) / GIB:.4f} GiB, more than 1.15x the "
        f"one-fresh-copy bound {(fixture_w13 + 2 * one_expert) / GIB:.4f} GiB; "
        "the projection below no longer holds"
    )
    assert residuals[-1] <= 0, "repack leaves extra bytes resident; projection invalid"

    def gib(n):
        return n / GIB

    w13 = RVN_EXPERTS * (2 * RVN_INTERMEDIATE) * (RVN_HIDDEN // 2)
    w2 = RVN_EXPERTS * RVN_INTERMEDIATE * (RVN_HIDDEN // 2)
    scales = RVN_EXPERTS * (
        2 * RVN_INTERMEDIATE * (RVN_HIDDEN // RVN_GROUP)
        + RVN_HIDDEN * (RVN_INTERMEDIATE // RVN_GROUP)
    )
    # Cross-check the fixture's arithmetic against the real shard headers:
    # 63.282 GiB of expert tensors, of which 7.031 GiB are fp8 scales.
    assert abs(gib(RVN_LAYERS * (w13 + w2 + scales)) - 63.282) < 0.02
    assert abs(gib(RVN_LAYERS * scales) - 7.031) < 0.02

    peak = w13 + 2 * (w13 // RVN_EXPERTS)
    experts = RVN_LAYERS * (w13 + w2 + scales)
    with_fix = gib(experts + peak) + NON_EXPERT_GIB + CONTEXT_GIB
    without_fix = with_fix + gib(RVN_LAYERS * scales)
    assert without_fix - with_fix == pytest.approx(gib(RVN_LAYERS * scales))
    assert with_fix <= BUDGET_GIB, (
        f"48-layer marlin projection needs {with_fix:.2f} GiB, over the "
        f"{BUDGET_GIB:.0f} GiB budget"
    )
    # The headroom 0053 buys is the whole difference the launch was short by.
    assert BUDGET_GIB - with_fix >= 8.0, (
        f"fixed projection leaves only {BUDGET_GIB - with_fix:.2f} GiB of the "
        f"{BUDGET_GIB:.0f} GiB budget"
    )

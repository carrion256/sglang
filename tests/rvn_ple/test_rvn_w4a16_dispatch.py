"""WP1: patches/0048-rvn-w4a16-dispatch.patch — RVN W4A16_NVFP4 dispatch.

The RVN checkpoint (qwen38-flash-next-uncensored) declares
``quant_algo: "W4A16_NVFP4"``. Serving it must recognize that explicit format
string and run packed E2M1 experts (group-16 E4M3 block scales, per-expert
FP32 gate/up-shared and down-independent global scales, BF16 activations)
through the Marlin MoE path — never a silent W4A4 downgrade.

Conventions (repo style + provenance attestation):
- Preimage = deployed tree bytes, read from /sgl-workspace/sglang (container)
  or /tmp/rvn-preimage-sglang (extracted), NEVER runtime/.
- Code under test is loaded with importlib.util.spec_from_file_location on
  single files with stubbed sglang modules; never ``import sglang``.
- The patch is applied into a scratch tree assembled from the preimage before
  the patched module is loaded.
"""

import hashlib
import importlib.util
import pathlib
import re as _re_stdlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_PATH = ROOT / "patches" / "0048-rvn-w4a16-dispatch.patch"
PREIMAGE_SHA256 = (
    "b05ed81bef0443781c79c7a6daf05204b8d6c50903cd84dae87ef3303c93d311"
)

_PREIMAGE_LAYOUTS = [
    (pathlib.Path("/sgl-workspace/sglang"), pathlib.Path("python/sglang")),
    (pathlib.Path("/tmp/rvn-preimage-sglang"), pathlib.Path(".")),
]
_MODELFILE_DEPLOYED = pathlib.Path("srt/layers/quantization/modelopt_quant.py")
_MODELFILE_MC = pathlib.Path("srt/configs/model_config.py")


def _env_override():
    raw = None
    import os

    raw = os.environ.get("RVN_W4A16_PREIMAGE_ROOT")
    if raw:
        root = pathlib.Path(raw)
        return root, pathlib.Path(".")
    return None


def preimage_root():
    candidates = []
    override = _env_override()
    if override:
        candidates.append(override)
    candidates.extend(_PREIMAGE_LAYOUTS)
    for root, pkg in candidates:
        if (root / pkg / _MODELFILE_DEPLOYED).is_file():
            return root, pkg
    return None, None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Verbatim checkpoint dicts (capture of the deployed files; keys/values are
# what the routing chain actually indexes).
# ---------------------------------------------------------------------------
RVN_EXCLUDE_MODULES = [
    "lm_head",
    "*embed_tokens*",
    "*linear_attn*",
    "*self_attn*",
    "*attn_hyper_connection*",
    "*mlp_hyper_connection*",
    "*hyper_connection_mixer*",
    "*ple*",
    "*shared_expert*",
    "*mlp.gate*",
    "*norm*",
]

# /models/qwen38-flash-next-uncensored/config.json:quantization_config
RVN_BLOB = {
    "quant_method": "modelopt",
    "producer": {"name": "rvn-flashinfer-weight-only-rtn", "version": "1"},
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": RVN_EXCLUDE_MODULES,
    },
}

# /models/qwen38-flash-next-uncensored/hf_quant_config.json (flat)
RVN_FLAT = {
    "quant_algo": "W4A16_NVFP4",
    "kv_cache_quant_algo": None,
    "group_size": 16,
    "exclude_modules": RVN_EXCLUDE_MODULES,
}

# LIL (qwen38-flash-next) mixed-precision shape: per-layer map, mtp experts
# already carry W4A16_NVFP4 and must keep their current dispatch (opt-in rule).
LIL_MIXED_BLOB = {
    "quant_method": "modelopt_mixed",
    "quant_algo": "MIXED_PRECISION",
    "group_size": 16,
    "exclude_modules": [],
    "quantized_layers": {
        "model.language_model.layers.0.linear_attn.in_proj_qkv": {
            "quant_algo": "MXFP8",
            "group_size": 32,
        },
        "model.language_model.layers.0.mlp.experts": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        },
        "model.language_model.layers.0.mlp.gate_proj": {
            "quant_algo": "MXFP8",
            "group_size": 32,
        },
        "model.visual.blocks.0.mlp.linear_fc2": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
        "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    },
}

# LIL qwen38-flash-next config.json:quantization_config (real shape:
# quant_method "modelopt", quant_algo "MIXED_PRECISION", quantized_layers).
LIL_MIXED_BLOB_REAL = {
    "quant_method": "modelopt",
    "producer": {"name": "modelopt", "version": "0.0"},
    "quant_algo": "MIXED_PRECISION",
    "group_size": 16,
    "quantized_layers": {
        "model.language_model.layers.0.mlp.experts": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        },
        "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        "mtp.layers.48.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    },
}

W4A4_BLOB = {
    "quant_method": "modelopt",
    "producer": {"name": "modelopt", "version": "0.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": ["lm_head"],
    },
}

LEGACY_FORMAT_STRINGS = [
    "FP8",
    "NVFP4",
    "NVFP4_AWQ",
    "W8A16",
    "W4A16",  # not the magic string: must keep the legacy rejection
    "FP4",
    "MXFP8",
    "GPTQ",
    "",
]


# ---------------------------------------------------------------------------
# Stub harness: every sglang symbol modelopt_quant.py / model_config.py
# import is provided by these fakes; loaded modules see identical
# environments, so preimage-vs-patched deltas isolate to the patch.
# ---------------------------------------------------------------------------
class _EnvVal:
    def __init__(self, values, key):
        self._values = values
        self._key = key

    def get(self):
        return self._values.get(self._key, None)


class _StubEnvs:
    def __init__(self):
        self.values = {}

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return _EnvVal(self.values, key)


class _BackendState:
    def __init__(self):
        self.backend = None
        self.is_cuda = True
        self.capability = (12, 0)
        self.blackwell = True


STUB_ENVS = _StubEnvs()
STUB_BACKEND = _BackendState()


def _make_backend_enum():
    import enum

    class MoeRunnerBackend(enum.Enum):
        AUTO = "auto"
        DEEP_GEMM = "deep_gemm"
        TRITON = "triton"
        FLASHINFER_TRTLLM = "flashinfer_trtllm"
        FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"
        FLASHINFER_CUTLASS = "flashinfer_cutlass"
        FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
        CUTLASS = "cutlass"
        MARLIN = "marlin"

        def _is(self, *names):
            return self.name in names

        def is_auto(self):
            return self._is("AUTO")

        def is_marlin(self):
            return self._is("MARLIN")

        def is_flashinfer_trtllm(self):
            return self._is("FLASHINFER_TRTLLM")

        def is_flashinfer_trtllm_routed(self):
            return self._is("FLASHINFER_TRTLLM_ROUTED")

        def is_flashinfer_cutlass(self):
            return self._is("FLASHINFER_CUTLASS")

        def is_flashinfer_cutedsl(self):
            return self._is("FLASHINFER_CUTEDSL")

        def is_cutlass(self):
            return self._is("CUTLASS")

    MoeRunnerBackend.AUTO  # noqa: B018 - force member resolution
    return MoeRunnerBackend


class _Capturing:
    """Base for stub payloads: records constructor kwargs."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_stubs():
    if getattr(_install_stubs, "_done", False):
        return
    MOE_BACKEND = _make_backend_enum()
    STUB_BACKEND.backend = MOE_BACKEND.AUTO

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _Any:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return None

    def identity_decorator(*args, **kwargs):
        def deco(func):
            return func

        return deco

    module("sglang.srt.layers.moe.token_dispatcher")
    sys.modules["sglang.srt.layers.moe.token_dispatcher"].StandardDispatchOutput = _Any
    sys.modules["sglang.srt.layers.moe.token_dispatcher"].StandardCombineOutput = _Any
    module(
        "sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl",
        FlashInferCuteDSLConfig=_Any,
        flashinfer_cutedsl_moe=lambda *a, **k: None,
    )
    module(
        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass",
        FlashInferCutlassMoEMethod=_Any,
    )
    module(
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        FlashInferTrtllmBackendsNoRouteMoEMethod=_Any,
        FlashInferTrtllmBackendsRoutedMoEMethod=_Any,
    )

    class QuantizationConfig:
        def __init__(self):
            pass

        @staticmethod
        def get_from_keys(config, keys):
            for key in keys:
                if key in config:
                    return config[key]
            raise ValueError(
                f"Cannot find any of {keys} in the model's quantization config."
            )

        @classmethod
        def _modelopt_override_quantization_method(cls, hf_quant_config, user_quant):
            # Faithful transcription of the deployed base_config classmethod.
            from sglang.srt.configs.model_config import REQUANTIZATION_METHODS

            if user_quant == "nvfp4_online" or user_quant in REQUANTIZATION_METHODS:
                return None
            quant_algo = hf_quant_config.get("quant_algo", "").upper()
            if user_quant == "modelopt":
                if quant_algo == "MXFP8":
                    return "mxfp8"
                elif quant_algo == "FP8":
                    return "modelopt_fp8"
                elif "NVFP4" in quant_algo or "FP4" in quant_algo:
                    return "modelopt_fp4"
            if hf_quant_config.get("quant_method", "") == "modelopt_fp8":
                return "modelopt_fp8"
            elif hf_quant_config.get("quant_method", "") == "modelopt_fp4":
                return "modelopt_fp4"
            return None

    class _Layer:
        pass

    class VocabParallelEmbedding(_Layer):
        pass

    class ParallelLMHead(VocabParallelEmbedding):
        pass

    module("sglang")
    module("sglang.kernels")
    module("sglang.kernels.ops")
    module("sglang.kernels.ops.quantization")
    module(
        "sglang.kernels.ops.quantization.fp8_kernel",
        scaled_fp8_quant=lambda *a, **k: None,
    )
    module("sglang.srt")
    module("sglang.srt.environ", envs=STUB_ENVS)
    module("sglang.srt.configs")
    module("sglang.srt.configs.model_config", REQUANTIZATION_METHODS=["quark_mxfp4"])
    module("sglang.srt.layers")
    module(
        "sglang.srt.layers.moe",
        MoeRunner=type("MoeRunner", (_Capturing,), {"run": lambda self, *a, **k: None}),
        MoeRunnerBackend=MOE_BACKEND,
        MoeRunnerConfig=_Capturing,
        get_moe_runner_backend=lambda: STUB_BACKEND.backend,
    )
    module("sglang.srt.layers.moe.moe_runner")
    module(
        "sglang.srt.layers.moe.moe_runner.triton",
        TritonMoeQuantInfo=type("TritonMoeQuantInfo", (_Capturing,), {}),
    )
    module(
        "sglang.srt.layers.moe.moe_runner.marlin",
        MarlinMoeQuantInfo=type("MarlinMoeQuantInfo", (_Capturing,), {}),
    )
    module(
        "sglang.srt.layers.moe.fused_moe_triton",
        FusedMoE=type("FusedMoE", (_Layer,), {}),
        FusedMoeWeightScaleSupported=type(
            "FusedMoeWeightScaleSupported",
            (),
            {
                "BLOCK": types.SimpleNamespace(value="block"),
                "TENSOR": types.SimpleNamespace(value="tensor"),
            },
        ),
    )
    module(
        "sglang.srt.layers.moe.utils",
        is_flashinfer_cutedsl_v1_path=lambda *a, **k: False,
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda *a, **k: False,
    )
    module(
        "sglang.srt.layers.parameter",
        ModelWeightParameter=_Capturing,
        PerTensorScaleParameter=_Capturing,
    )
    module("sglang.srt.layers.linear", LinearBase=type("LinearBase", (_Layer,), {}))
    module(
        "sglang.srt.layers.vocab_parallel_embedding",
        VocabParallelEmbedding=VocabParallelEmbedding,
        ParallelLMHead=ParallelLMHead,
    )
    module("sglang.srt.layers.quantization")
    sys.modules["sglang.srt.layers.quantization"].QUANTIZATION_METHODS = {}
    module(
        "sglang.srt.layers.quantization.base_config",
        QuantizationConfig=QuantizationConfig,
        QuantizeMethodBase=type("QuantizeMethodBase", (), {"__init__": lambda self, *a, **k: None}),
        LinearMethodBase=type("LinearMethodBase", (), {"__init__": lambda self, *a, **k: None}),
        FusedMoEMethodBase=type(
            "FusedMoEMethodBase",
            (),
            {"__init__": lambda self, *a, **k: None},
        ),
    )
    module(
        "sglang.srt.layers.quantization.fp4_utils",
        fp4_quantize=lambda *a, **k: None,
        get_fp4_gemm_runner_backend=lambda: types.SimpleNamespace(
            is_marlin=lambda: False,
            is_flashinfer_trtllm=lambda: False,
            is_flashinfer_cutlass=lambda: False,
            get_flashinfer_backend=lambda: "trtllm",
        ),
    )
    module(
        "sglang.srt.layers.quantization.fp8",
        Fp8Config=type("Fp8Config", (), {"__init__": lambda self, *a, **k: None}),
        Fp8LinearMethod=_Any,
        Fp8MoEMethod=_Any,
    )
    module(
        "sglang.srt.layers.quantization.fp8_utils",
        apply_fp8_linear=lambda *a, **k: None,
        apply_fp8_linear_bmm_flashinfer=lambda *a, **k: None,
        can_auto_enable_marlin_fp8=lambda *a, **k: False,
        cutlass_fp8_supported=lambda *a, **k: False,
        flashinfer_per_tensor_fp8_supported=lambda *a, **k: False,
        is_blackwell_supported=lambda *a, **k: STUB_BACKEND.blackwell,
    )
    module(
        "sglang.srt.layers.quantization.kv_cache",
        BaseKVCacheMethod=type(
            "BaseKVCacheMethod", (), {"__init__": lambda self, *a, **k: None}
        ),
    )

    def _record_prepare(layer, *args, **kwargs):
        layer.marlin_prepared = True
        return layer

    module(
        "sglang.srt.layers.quantization.marlin_utils_fp4",
        apply_fp4_marlin_linear=lambda *a, **k: None,
        prepare_moe_nvfp4_layer_for_marlin=_record_prepare,
        prepare_nvfp4_layer_for_marlin=_record_prepare,
    )
    module(
        "sglang.srt.layers.quantization.marlin_utils_fp8",
        prepare_fp8_layer_for_marlin=_record_prepare,
    )
    module(
        "sglang.srt.layers.quantization.unquant",
        UnquantizedLinearMethod=type(
            "UnquantizedLinearMethod", (), {"__init__": lambda self, *a, **k: None}
        ),
    )
    module(
        "sglang.srt.layers.quantization.utils",
        convert_to_channelwise=lambda x, *a, **k: x,
        is_layer_skipped=lambda prefix, excluded, mapping=None: False,
        per_tensor_dequantize=lambda *a, **k: None,
        requantize_with_max_scale=lambda x, *a, **k: x,
        swizzle_blockscale=lambda x, *a, **k: x,
    )
    module(
        "sglang.srt.layers.quantization.vision_mxfp8",
        VisionMxfp8PaddedLinearMethod=_Any,
        VisionNvFp4A16LinearMethod=_Any,
    )
    module(
        "sglang.srt.layers.radix_attention",
        RadixAttention=type("RadixAttention", (_Layer,), {}),
    )
    module(
        "sglang.srt.layers.utils",
        alias_or_bind_derived_param=lambda layer, target, source, value: setattr(
            layer, target, value
        ),
        copy_or_rebind_param=lambda layer, name, value: setattr(layer, name, value),
    )
    module("sglang.srt.utils")
    module(
        "sglang.srt.utils.common",
        get_device_capability=lambda *a, **k: STUB_BACKEND.capability,
        is_cuda=lambda *a, **k: STUB_BACKEND.is_cuda,
        is_sm120_supported=lambda *a, **k: True,
        round_up=lambda x, m: ((x + m - 1) // m) * m,
        set_weight_attrs=lambda d, attrs: d.update(attrs),
    )
    module(
        "sglang.srt.utils.custom_op",
        register_custom_op=identity_decorator,
    )
    module(
        "sglang.srt.utils.patch_torch",
        register_fake_if_exists=identity_decorator,
    )
    module("sglang.srt.models", utils=None)
    sys.modules["sglang.kernels.ops.quantization"].fp8_kernel = sys.modules[
        "sglang.kernels.ops.quantization.fp8_kernel"
    ]
    sys.modules["sglang.srt.layers.moe"].moe_runner = sys.modules[
        "sglang.srt.layers.moe.moe_runner"
    ]
    if "regex" not in sys.modules:
        sys.modules["regex"] = _re_stdlib
    # Force the module-under-test flashinfer fallback (try/except ImportError).
    sys.modules.setdefault("flashinfer", None)
    _install_stubs._done = True


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ROOT, _PKG = preimage_root()
PREIMAGE_MODELFILE = None
if _ROOT is not None:
    PREIMAGE_MODELFILE = _ROOT / _PKG / _MODELFILE_DEPLOYED

SCRATCH = None
PATCHED_MODELFILE = None
APPLY_CHECK_RC = None
APPLY_RC = None

_MODULES_BEFORE = set(sys.modules)


def tearDownModule():
    # Never leak stub modules or scratch trees into sibling test modules.
    for name in [key for key in sys.modules if key not in _MODULES_BEFORE]:
        del sys.modules[name]
    if SCRATCH is not None:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def _assemble_scratch_and_patch():
    global SCRATCH, PATCHED_MODELFILE, APPLY_CHECK_RC, APPLY_RC
    if PATCHED_MODELFILE is not None:
        return
    SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="rvn0048-"))
    target = SCRATCH / "python/sglang/srt/layers/quantization/modelopt_quant.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(PREIMAGE_MODELFILE, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != hashlib.sha256(
        PREIMAGE_MODELFILE.read_bytes()
    ).hexdigest():
        raise AssertionError("scratch preimage copy diverged from attested preimage")
    check = subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)],
        cwd=SCRATCH,
        capture_output=True,
        text=True,
    )
    APPLY_CHECK_RC = check.returncode
    if check.returncode != 0:
        raise AssertionError(f"git apply --check failed: {check.stderr}")
    apply = subprocess.run(
        ["git", "apply", str(PATCH_PATH)],
        cwd=SCRATCH,
        capture_output=True,
        text=True,
    )
    APPLY_RC = apply.returncode
    if apply.returncode != 0:
        raise AssertionError(f"git apply failed: {apply.stderr}")
    compile(target.read_text(), str(target), "exec")
    PATCHED_MODELFILE = target


def load_preimage():
    _install_stubs()
    if getattr(load_preimage, "_mod", None) is None:
        load_preimage._mod = _load_file(PREIMAGE_MODELFILE, "rvn0048_preimage_modelopt")
    return load_preimage._mod


def load_patched():
    _install_stubs()
    _assemble_scratch_and_patch()
    if getattr(load_patched, "_mod", None) is None:
        load_patched._mod = _load_file(PATCHED_MODELFILE, "rvn0048_patched_modelopt")
    return load_patched._mod


def load_model_config():
    _install_stubs()
    if getattr(load_model_config, "_mod", None) is not None:
        return load_model_config._mod
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except Exception:  # pragma: no cover - host without transformers
            stub = types.ModuleType("transformers")
            stub.PretrainedConfig = type("PretrainedConfig", (), {})
            sys.modules["transformers"] = stub
    if "sglang.srt.configs.embedding_model_spec" not in sys.modules:
        module_names = [
            "sglang.srt.configs.embedding_model_spec",
            "sglang.srt.configs.linear_attn_model_registry",
            "sglang.srt.server_args",
            "sglang.srt.utils.hf_transformers_utils",
            "sglang.srt.utils.runai_utils",
            "sglang.utils",
        ]
        for name in module_names:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        sys.modules[
            "sglang.srt.configs.embedding_model_spec"
        ].resolve_embedding_model_spec = lambda *a, **k: None
        sys.modules[
            "sglang.srt.configs.linear_attn_model_registry"
        ].get_linear_attn_config = lambda *a, **k: None
        sys.modules["sglang.srt.server_args"].ServerArgs = type("ServerArgs", (), {})
        hf_utils = sys.modules["sglang.srt.utils.hf_transformers_utils"]
        for fn in (
            "get_config",
            "get_context_length",
            "get_generation_config",
            "get_hf_text_config",
            "get_sparse_attention_config",
        ):
            setattr(hf_utils, fn, lambda *a, **k: None)
        runai = sys.modules["sglang.srt.utils.runai_utils"]
        runai.ObjectStorageModel = type("ObjectStorageModel", (), {})
        runai.is_runai_obj_uri = lambda *a, **k: False
        sys.modules["sglang.utils"].is_in_ci = lambda *a, **k: False
    sutils = sys.modules["sglang.srt.utils"]
    if not hasattr(sutils, "is_hip"):
        sutils.is_hip = lambda *a, **k: False
        sutils.is_sm100_supported = lambda *a, **k: False
        sutils.retry = lambda func=None, **k: (func if func is not None else (lambda f: f))
    mc_path = _ROOT / _PKG / _MODELFILE_MC
    load_model_config._mod = _load_file(mc_path, "rvn0048_model_config")
    return load_model_config._mod


def chain_resolve_quantization(quant_cfg, user_quant, modelopt_mod):
    """Run the real routing functions over a verbatim checkpoint dict."""
    mc = load_model_config()
    qm = sys.modules["sglang.srt.layers.quantization"].QUANTIZATION_METHODS
    qm.clear()
    qm.update(
        {
            "modelopt_mixed": modelopt_mod.ModelOptMixedPrecisionConfig,
            "modelopt_fp4": modelopt_mod.ModelOptFp4Config,
        }
    )

    class _Cfg:
        quantization_config = quant_cfg
        text_config = None

    class _Self:
        hf_config = _Cfg()
        model_path = "/nonexistent-rvn-checkpoint"
        quantization = user_quant
        is_draft_model = False
        is_draft_quantization_explicit = False
        _parse_quant_hf_config = mc.ModelConfig._parse_quant_hf_config
        _parse_modelopt_quant_config = mc.ModelConfig._parse_modelopt_quant_config
        _find_quant_modelslim_config = mc.ModelConfig._find_quant_modelslim_config

    inst = _Self()
    inst._parse_quant_hf_config()
    mc.ModelConfig._verify_quantization(inst)
    return inst.quantization


def moe_runner_selection(module, config, backend_name):
    """Resolve ModelOptNvFp4FusedMoEMethod's MoE runner under a backend."""
    MOE = sys.modules["sglang.srt.layers.moe"]
    STUB_BACKEND.backend = MOE.MoeRunnerBackend[backend_name]
    try:
        method = module.ModelOptNvFp4FusedMoEMethod(config)
    except Exception as exc:  # noqa: BLE001 - resolution tables capture raises
        return ("raise", type(exc).__name__)
    layer = types.SimpleNamespace(moe_runner_config=types.SimpleNamespace(is_gated=True))
    method.create_moe_runner(layer, MOE.MoeRunnerConfig(activation="silu"))
    return ("ok", method._moe_runner_backend.name)


def from_config_row(module, algo, shape):
    """Resolution row for ModelOptFp4Config.from_config on one format string."""
    if shape == "nested":
        cfg = {
            "quant_method": "modelopt",
            "producer": {"name": "modelopt", "version": "0"},
            "quantization": {
                "quant_algo": algo,
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            },
        }
    else:
        cfg = {
            "quant_algo": algo,
            "group_size": 16,
            "ignore": ["lm_head"],
        }
    try:
        cfg_obj = module.ModelOptFp4Config.from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - resolution tables capture raises
        return ("raise", type(exc).__name__)
    return (
        "ok",
        type(cfg_obj).__name__,
        cfg_obj.is_checkpoint_nvfp4_serialized,
        cfg_obj.is_awq,
        cfg_obj.group_size,
        cfg_obj.use_per_token_activation,
    )


def _mixed_config(module):
    cfg = module.ModelOptMixedPrecisionConfig.from_config(dict(LIL_MIXED_BLOB))
    return cfg


class TestPatchIntegrity(unittest.TestCase):
    def test_patch_exists_and_applies_to_preimage_scratch(self):
        if PREIMAGE_MODELFILE is None:
            self.skipTest("deployed preimage tree not available")
        self.assertTrue(PATCH_PATH.is_file(), f"missing {PATCH_PATH}")
        _assemble_scratch_and_patch()
        self.assertEqual(APPLY_CHECK_RC, 0)
        self.assertEqual(APPLY_RC, 0)
        preimage_bytes = PREIMAGE_MODELFILE.read_bytes()
        patched_bytes = PATCHED_MODELFILE.read_bytes()
        self.assertNotEqual(preimage_bytes, patched_bytes)

    def test_preimage_sha_matches_attested_export(self):
        if PREIMAGE_MODELFILE is None:
            self.skipTest("deployed preimage tree not available")
        self.assertEqual(_sha256(PREIMAGE_MODELFILE), PREIMAGE_SHA256)


class TestW4A16Recognition(unittest.TestCase):
    def setUp(self):
        if PREIMAGE_MODELFILE is None:
            self.skipTest("deployed preimage tree not available")
        _install_stubs()
        STUB_BACKEND.blackwell = True
        STUB_BACKEND.backend = sys.modules["sglang.srt.layers.moe"].MoeRunnerBackend.AUTO
        self.mod = load_patched()

    def test_rvn_blob_routes_to_modelopt_fp4(self):
        resolved = chain_resolve_quantization(dict(RVN_BLOB), "modelopt_mixed", self.mod)
        self.assertEqual(resolved, "modelopt_fp4")

    def test_rvn_blob_preimage_stays_modelopt_mixed(self):
        # Documents the pre-fix routing; the loader then dies in
        # ModelOptMixedPrecisionConfig.from_config with
        # "only supports MIXED_PRECISION checkpoints."
        resolved = chain_resolve_quantization(
            dict(RVN_BLOB), "modelopt_mixed", load_preimage()
        )
        self.assertEqual(resolved, "modelopt_mixed")

    def test_lil_mixed_blob_routing_unchanged(self):
        for module in (load_preimage(), self.mod):
            with self.subTest(module=module.__name__):
                resolved = chain_resolve_quantization(
                    dict(LIL_MIXED_BLOB), "modelopt_mixed", module
                )
                self.assertEqual(resolved, "modelopt_mixed")

    def test_w4a16_from_config_nested_and_flat_recognition(self):
        nested = self.mod.ModelOptFp4Config.from_config(
            {**RVN_BLOB, "packed_modules_mapping": None}
        )
        self.assertTrue(nested.is_w4a16_nvfp4)
        self.assertEqual(nested.quant_format, "W4A16_NVFP4")
        self.assertEqual(nested.group_size, 16)
        self.assertEqual(nested.exclude_modules, RVN_EXCLUDE_MODULES)
        flat = self.mod.ModelOptFp4Config.from_config(dict(RVN_FLAT))
        self.assertTrue(flat.is_w4a16_nvfp4)
        self.assertEqual(flat.exclude_modules, RVN_EXCLUDE_MODULES)
        with self.assertRaises(ValueError):
            self.mod.ModelOptFp4Config.from_config(
                {
                    "quant_algo": "W4A4_NVFP4",
                    "group_size": 16,
                    "ignore": ["lm_head"],
                }
            )

    def test_w4a16_expert_scale_semantics_preserved_on_config(self):
        import torch

        cfg = self.mod.ModelOptFp4Config.from_config(
            {**RVN_BLOB, "packed_modules_mapping": None}
        )
        self.assertEqual(cfg.weight_scale_dtype, torch.float8_e4m3fn)
        self.assertEqual(cfg.group_size, 16)
        self.assertEqual(cfg.expert_global_scale_dtype, torch.float32)
        self.assertTrue(cfg.expert_global_scale_shared_gate_up)
        self.assertTrue(cfg.expert_global_scale_down_independent)
        self.assertEqual(cfg.activation_dtype, torch.bfloat16)
        self.assertFalse(cfg.input_scale_calibrated)
        self.assertEqual(cfg.reconstruction, "bf16_direct")


    def test_preimage_rejects_rvn_uniform_dicts(self):
        # Pin the pre-fix rejection the RVN baseline hit (WP1 failure mode).
        pre = load_preimage()
        for shape, cfg_dict in (
            ("nested", {**RVN_BLOB, "packed_modules_mapping": None}),
            ("flat", dict(RVN_FLAT)),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    pre.ModelOptFp4Config.from_config(cfg_dict)

    def test_reroute_gate_requires_uniform_checkpoint(self):
        # A quantized_layers map must never be hijacked out of the mixed
        # path, even when a layer entry is W4A16_NVFP4 (LIL mtp experts).
        mixed_shaped = {
            "quant_method": "modelopt",
            "producer": {"name": "modelopt", "version": "1"},
            "quantization": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "quantized_layers": {
                "mtp.layers.0.mlp.experts": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
            },
        }
        for name, cfg_dict in (
            ("mixed_shaped", mixed_shaped),
            ("lil_real", dict(LIL_MIXED_BLOB_REAL)),
        ):
            for module in (load_preimage(), self.mod):
                with self.subTest(dict=name, module=module.__name__):
                    self.assertIsNone(
                        module.ModelOptMixedPrecisionConfig.override_quantization_method(
                            dict(cfg_dict), "modelopt_mixed"
                        )
                    )
        self.assertIsNone(
            load_preimage().ModelOptMixedPrecisionConfig.override_quantization_method(
                dict(RVN_BLOB), "modelopt_mixed"
            )
        )
        self.assertEqual(
            self.mod.ModelOptMixedPrecisionConfig.override_quantization_method(
                dict(RVN_BLOB), "modelopt_mixed"
            ),
            "modelopt_fp4",
        )

    def test_rvn_exclusions_resolve_to_unquantized(self):
        # RVN export receipt: excluded linears fall back to unquantized and
        # non-excluded modules take the W4A16 dispatch. down_proj is the
        # positive control (no RVN glob covers it, so the A16 method must
        # fire); the embed row pins the embed branch's no-quant-method
        # resolution (embeddings are unquantized regardless of the glob),
        # and the excluded-FusedMoE row pins the None fallback.
        cfg = self.mod.ModelOptFp4Config.from_config(
            {**RVN_BLOB, "packed_modules_mapping": None}
        )
        linear_stub = sys.modules["sglang.srt.layers.linear"].LinearBase
        moe_stub = sys.modules["sglang.srt.layers.moe.fused_moe_triton"].FusedMoE
        embed_stub = sys.modules[
            "sglang.srt.layers.vocab_parallel_embedding"
        ].VocabParallelEmbedding

        class _Linear(linear_stub):
            pass

        class _MoE(moe_stub):
            pass

        class _Embed(embed_stub):
            pass

        unquantized = self.mod.UnquantizedLinearMethod
        cases = [
            ("model.layers.0.mlp.ple_table", _Linear(), unquantized),
            ("lm_head", _Linear(), unquantized),
            ("model.layers.0.self_attn.q_proj", _Linear(), unquantized),
            ("model.embed_tokens", _Embed(), None),
            ("model.layers.0.mlp.shared_expert.gate_proj", _Linear(), unquantized),
            ("model.layers.0.mlp.gate", _Linear(), unquantized),
            (
                "model.layers.0.mlp.experts",
                _MoE(),
                self.mod.ModelOptNvFp4FusedMoEMethod,
            ),
            (
                "model.layers.0.mlp.down_proj",
                _Linear(),
                self.mod.ModelOptNvFp4A16LinearMethod,
            ),
            ("model.layers.0.mlp.gate_up_proj.experts", _MoE(), None),
        ]
        for prefix, layer, expected in cases:
            with self.subTest(prefix=prefix):
                method = cfg.get_quant_method(layer, prefix)
                if expected is None:
                    self.assertIsNone(method)
                else:
                    self.assertIs(type(method), expected)

class TestW4A16MarlinDispatch(unittest.TestCase):
    def setUp(self):
        if PREIMAGE_MODELFILE is None:
            self.skipTest("deployed preimage tree not available")
        _install_stubs()
        STUB_BACKEND.blackwell = True
        STUB_BACKEND.backend = sys.modules["sglang.srt.layers.moe"].MoeRunnerBackend.AUTO
        self.mod = load_patched()
        self.pre = load_preimage()
        import torch

        self.torch = torch

    def _w4a16_config(self, module):
        return module.ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=["lm_head"],
            quant_format="W4A16_NVFP4",
        )

    def test_w4a16_auto_backend_selects_marlin(self):
        STUB_BACKEND.blackwell = True
        self.assertEqual(
            moe_runner_selection(self.mod, self._w4a16_config(self.mod), "AUTO"),
            ("ok", "MARLIN"),
        )
        self.assertEqual(
            moe_runner_selection(self.mod, self._w4a16_config(self.mod), "MARLIN"),
            ("ok", "MARLIN"),
        )

    def test_w4a16_preimage_silently_downgrades_to_w4a4(self):
        # Evidence of the bug being fixed: the preimage cannot even express
        # W4A16 (no quant_format), and a byte-identical serialized expert
        # config resolves to the W4A4 TRT-LLM runner on Blackwell-class HW.
        STUB_BACKEND.blackwell = True
        pre_cfg = self.pre.ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=["lm_head"],
        )
        self.assertEqual(
            moe_runner_selection(self.pre, pre_cfg, "AUTO"),
            ("ok", "FLASHINFER_TRTLLM"),
        )

    def test_marlin_pin_is_cross_site_consistent(self):
        # Partial override is the bug class: every consumer of the backend
        # (create_weights / process_weights_after_loading / apply read
        # getattr(self, "_moe_runner_backend", get_moe_runner_backend()))
        # must see the same pinned Marlin value, starting at __init__.
        MOE = sys.modules["sglang.srt.layers.moe"]
        cfg = self._w4a16_config(self.mod)
        method = self.mod.ModelOptNvFp4FusedMoEMethod(cfg)
        self.assertEqual(
            getattr(method, "_moe_runner_backend", "MISSING").name, "MARLIN"
        )
        self.assertFalse(method.enable_flashinfer_trtllm_moe)
        self.assertFalse(method.enable_flashinfer_cutlass_moe)
        self.assertFalse(method.enable_flashinfer_cutedsl_moe)
        layer = types.SimpleNamespace(
            moe_runner_config=types.SimpleNamespace(is_gated=True)
        )
        method.create_moe_runner(layer, MOE.MoeRunnerConfig(activation="silu"))
        self.assertEqual(method._moe_runner_backend.name, "MARLIN")
        self.assertEqual(method.runner.args[0].name, "MARLIN")

    def test_w4a16_rejects_conflicting_w4a4_runner_selection(self):
        STUB_BACKEND.blackwell = True
        for backend in (
            "FLASHINFER_TRTLLM",
            "FLASHINFER_TRTLLM_ROUTED",
            "FLASHINFER_CUTLASS",
            "FLASHINFER_CUTEDSL",
            "CUTLASS",
        ):
            with self.subTest(backend=backend):
                selection = moe_runner_selection(
                    self.mod, self._w4a16_config(self.mod), backend
                )
                self.assertEqual(selection[0], "raise")
                self.assertEqual(selection[1], "ValueError")

    def test_w4a16_rejects_conflicting_per_token_activation(self):
        with self.assertRaises(ValueError):
            self.mod.ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                group_size=16,
                exclude_modules=["lm_head"],
                use_per_token_activation=True,
                quant_format="W4A16_NVFP4",
            )
        STUB_ENVS.values["SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION"] = True
        try:
            with self.assertRaises(ValueError):
                self.mod.ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    kv_cache_quant_algo=None,
                    group_size=16,
                    exclude_modules=["lm_head"],
                    quant_format="W4A16_NVFP4",
                )
        finally:
            STUB_ENVS.values.pop("SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION", None)

    def test_static_w4a4_metadata_alone_does_not_select_w4a16(self):
        STUB_BACKEND.blackwell = True
        for module in (self.pre, self.mod):
            with self.subTest(module=module.__name__):
                cfg = module.ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    kv_cache_quant_algo=None,
                    group_size=16,
                    exclude_modules=["lm_head"],
                    use_per_token_activation=False,
                )
                self.assertFalse(getattr(cfg, "is_w4a16_nvfp4", False))
                self.assertEqual(
                    moe_runner_selection(module, cfg, "AUTO"),
                    ("ok", "FLASHINFER_TRTLLM"),
                )

    def test_mixed_w4a16_layers_keep_existing_dispatch(self):
        # Opt-in rule: the LIL mixed path's nvfp4a16 config (mtp experts,
        # vision fc2) must keep its runner selection untouched.
        STUB_BACKEND.blackwell = True
        for module in (self.pre, self.mod):
            with self.subTest(module=module.__name__):
                cfg = _mixed_config(module)
                selection = moe_runner_selection(
                    module, cfg.nvfp4a16_config, "FLASHINFER_CUTLASS"
                )
                self.assertEqual(selection, ("ok", "FLASHINFER_CUTLASS"))

    def test_w4a16_linear_and_moe_method_selection(self):
        cfg = self._w4a16_config(self.mod)
        linear = types.SimpleNamespace(spec="linear")
        linear_stub = sys.modules["sglang.srt.layers.linear"].LinearBase
        moe_stub = sys.modules["sglang.srt.layers.moe.fused_moe_triton"].FusedMoE

        class _Linear(linear_stub):
            pass

        class _MoE(moe_stub):
            pass

        method = cfg.get_quant_method(_Linear(), "model.layers.0.mlp.gate_proj")
        self.assertIs(type(method), self.mod.ModelOptNvFp4A16LinearMethod)
        moe_method = cfg.get_quant_method(_MoE(), "model.layers.0.mlp.experts")
        self.assertIs(type(moe_method), self.mod.ModelOptNvFp4FusedMoEMethod)
        # Legacy NVFP4 config keeps the W4A4 linear method.
        w4a4 = self.mod.ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=["lm_head"],
            quant_format="NVFP4",
        )
        method4 = w4a4.get_quant_method(_Linear(), "model.layers.0.mlp.gate_proj")
        self.assertIs(type(method4), self.mod.ModelOptFp4LinearMethod)

    def test_marlin_payload_preserves_expert_scale_conventions(self):
        torch = self.torch
        cfg = self._w4a16_config(self.mod)
        STUB_BACKEND.blackwell = True
        STUB_BACKEND.backend = sys.modules["sglang.srt.layers.moe"].MoeRunnerBackend.AUTO
        method = self.mod.ModelOptNvFp4FusedMoEMethod(cfg)
        moe_stub_cfg = types.SimpleNamespace(is_gated=True, activation="silu")
        layer = types.SimpleNamespace(
            moe_runner_config=moe_stub_cfg,
            num_local_experts=2,
            w13_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
            w2_weight=torch.zeros(2, 4, 2, dtype=torch.uint8),
            w13_weight_scale=torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn),
            w2_weight_scale=torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn),
            w13_weight_scale_2=torch.tensor([[2.0, 2.0], [4.0, 4.0]], dtype=torch.float32),
            w2_weight_scale_2=torch.tensor([8.0, 16.0], dtype=torch.float32),
            dispatcher=None,
        )
        method.create_moe_runner(layer, moe_stub_cfg)
        method.process_weights_after_loading(layer)
        self.assertTrue(getattr(layer, "marlin_prepared", False))
        # gate/up collapse: one shared per-expert FP32 global scale.
        self.assertEqual(tuple(layer.w13_weight_scale_2.shape), (2,))
        torch.testing.assert_close(
            layer.w13_weight_scale_2, torch.tensor([2.0, 4.0]), rtol=0, atol=0
        )
        info = method.get_marlin_quant_info(layer)
        self.assertEqual(info.kwargs["weight_bits"], 4)
        self.assertIs(info.kwargs["w13_global_scale"], layer.w13_weight_scale_2)
        self.assertIs(info.kwargs["w2_global_scale"], layer.w2_weight_scale_2)
        self.assertIs(info.kwargs["w13_scales"], layer.w13_weight_scale)
        self.assertIs(info.kwargs["w2_scales"], layer.w2_weight_scale)

    def test_preimage_and_patched_marlin_collapse_identical_for_w4a4(self):
        # The scale-convention code is untouched: identical collapse for a
        # plain NVFP4 config driven through an explicit marlin backend.
        torch = self.torch
        results = {}
        for tag, module in (("pre", self.pre), ("post", self.mod)):
            cfg = module.ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                group_size=16,
                exclude_modules=["lm_head"],
            )
            STUB_BACKEND.blackwell = False
            STUB_BACKEND.backend = sys.modules[
                "sglang.srt.layers.moe"
            ].MoeRunnerBackend.MARLIN
            method = module.ModelOptNvFp4FusedMoEMethod(cfg)
            moe_stub_cfg = types.SimpleNamespace(is_gated=True, activation="silu")
            layer = types.SimpleNamespace(
                moe_runner_config=moe_stub_cfg,
                w13_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
                w2_weight=torch.zeros(2, 4, 2, dtype=torch.uint8),
                w13_weight_scale=torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn),
                w2_weight_scale=torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn),
                w13_weight_scale_2=torch.tensor(
                    [[2.0, 2.0], [4.0, 4.0]], dtype=torch.float32
                ),
                w2_weight_scale_2=torch.tensor([8.0, 16.0], dtype=torch.float32),
                dispatcher=None,
            )
            method.create_moe_runner(layer, moe_stub_cfg)
            method.process_weights_after_loading(layer)
            results[tag] = layer.w13_weight_scale_2
        torch.testing.assert_close(results["pre"], results["post"], rtol=0, atol=0)
        STUB_BACKEND.blackwell = True


class TestLegacyResolutionUnchanged(unittest.TestCase):
    """LIL regression guard: existing format strings resolve identically."""

    def setUp(self):
        if PREIMAGE_MODELFILE is None:
            self.skipTest("deployed preimage tree not available")
        _install_stubs()
        STUB_BACKEND.blackwell = True
        STUB_BACKEND.backend = sys.modules["sglang.srt.layers.moe"].MoeRunnerBackend.AUTO
        self.pre = load_preimage()
        self.post = load_patched()

    def test_from_config_tables_identical_for_existing_formats(self):
        for shape in ("nested", "flat"):
            for algo in LEGACY_FORMAT_STRINGS:
                with self.subTest(shape=shape, algo=algo):
                    self.assertEqual(
                        from_config_row(self.pre, algo, shape),
                        from_config_row(self.post, algo, shape),
                    )

    def test_override_routing_tables_identical_except_w4a16(self):
        quant_cfgs = {
            "lil-mixed": dict(LIL_MIXED_BLOB),
            "w4a4-nested": dict(W4A4_BLOB),
            "modelopt-fp4-method": {
                "quant_method": "modelopt_fp4",
                "quant_algo": "NVFP4",
            },
            "empty": {},
        }
        users = [None, "modelopt", "modelopt_fp4", "modelopt_fp8", "modelopt_mixed", "fp8"]
        for name, cfg in quant_cfgs.items():
            for user in users:
                with self.subTest(cfg=name, user=user):
                    self.assertEqual(
                        self.pre.ModelOptMixedPrecisionConfig.override_quantization_method(
                            dict(cfg), user
                        ),
                        self.post.ModelOptMixedPrecisionConfig.override_quantization_method(
                            dict(cfg), user
                        ),
                    )

    def test_mixed_resolution_and_method_tables_identical(self):
        prefixes = [
            ("model.language_model.layers.0.linear_attn.in_proj_qkv", "MXFP8"),
            ("model.language_model.layers.0.mlp.experts", "NVFP4"),
            ("model.visual.blocks.0.mlp.linear_fc2", "W4A16_NVFP4"),
            ("mtp.layers.0.mlp.experts", "W4A16_NVFP4"),
            ("model.language_model.layers.7.mlp.down_proj", None),
            ("totally.unknown.layer", None),
        ]
        linear_stub = sys.modules["sglang.srt.layers.linear"].LinearBase
        moe_stub = sys.modules["sglang.srt.layers.moe.fused_moe_triton"].FusedMoE
        embed_stub = sys.modules[
            "sglang.srt.layers.vocab_parallel_embedding"
        ].VocabParallelEmbedding
        attn_stub = sys.modules["sglang.srt.layers.radix_attention"].RadixAttention

        class _Linear(linear_stub):
            pass

        class _MoE(moe_stub):
            pass

        class _Embed(embed_stub):
            pass

        class _Attn(attn_stub):
            pass

        layers = {
            "linear": _Linear(),
            "moe": _MoE(),
            "embed": _Embed(),
            "attn": _Attn(),
        }
        cfgs = {"pre": _mixed_config(self.pre), "post": _mixed_config(self.post)}
        for prefix, expected_algo in prefixes:
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    cfgs["pre"]._resolve_quant_algo(prefix),
                    cfgs["post"]._resolve_quant_algo(prefix),
                )
                self.assertEqual(
                    cfgs["pre"]._resolve_quant_algo(prefix), expected_algo
                )
            for layer_name, layer in layers.items():
                with self.subTest(prefix=prefix, layer=layer_name):
                    pre_m = cfgs["pre"].get_quant_method(layer, prefix)
                    post_m = cfgs["post"].get_quant_method(layer, prefix)
                    self.assertEqual(type(pre_m).__name__, type(post_m).__name__)

    def test_w4a4_moe_runner_selection_identical(self):
        STUB_BACKEND.blackwell = True
        for algo in ("NVFP4", "NVFP4_AWQ"):
            for backend in ("AUTO", "MARLIN", "FLASHINFER_TRTLLM", "FLASHINFER_CUTLASS"):
                with self.subTest(algo=algo, backend=backend):
                    pre_cfg = self.pre.ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True,
                        kv_cache_quant_algo=None,
                        group_size=16,
                        exclude_modules=["lm_head"],
                        is_awq="AWQ" in algo,
                    )
                    post_cfg = self.post.ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True,
                        kv_cache_quant_algo=None,
                        group_size=16,
                        exclude_modules=["lm_head"],
                        is_awq="AWQ" in algo,
                    )
                    self.assertEqual(
                        moe_runner_selection(self.pre, pre_cfg, backend),
                        moe_runner_selection(self.post, post_cfg, backend),
                    )
        STUB_BACKEND.blackwell = True


if __name__ == "__main__":
    unittest.main()

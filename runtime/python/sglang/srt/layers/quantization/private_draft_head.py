"""Runtime-only private NEXTN NVFP4 head; target and embeddings stay BF16."""
import copy
import os


def private_head_shell(target):
    """Copy TP metadata but own all module registries before replacing weights."""
    private = copy.copy(target)
    private._parameters = target._parameters.copy()
    private._buffers = target._buffers.copy()
    private._modules = target._modules.copy()
    del private.weight
    return private


def select_draft_head(target):
    value = os.environ.get('SGLANG_PRIVATE_DRAFT_NVFP4_A16', '0')
    if value == '0':
        return target
    if value != '1':
        raise ValueError('SGLANG_PRIVATE_DRAFT_NVFP4_A16 must be 0 or 1')
    return build_private_head(target)


def build_private_head(target):
    """Quantize the already TP-sharded target weight without writing it."""
    import logging
    from types import SimpleNamespace

    import torch
    from flashinfer import SfLayout, nvfp4_quantize
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_nvfp4_layer_for_marlin,
    )
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4A16LinearMethod,
    )

    weight = target.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_cuda:
        raise ValueError('Private draft head requires a CUDA BF16 matrix')
    if weight.shape[0] < 2048:
        # Installed Marlin selects BF16 atomic reduction below this boundary;
        # restrict this trial to the non-atomic, vocabulary-sized head path.
        raise ValueError('Private draft head requires N >= 2048')
    if weight.shape[1] % 16 or getattr(target, 'bias', None) is not None:
        raise ValueError('Private draft head requires K divisible by 16 and no bias')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Private draft head must be initialized before graph capture')
    with torch.no_grad():
        amax = weight.abs().amax().float()
        if not torch.isfinite(amax):
            raise ValueError('Private draft head requires finite weights')
        scale = torch.where(amax > 0, amax / (448 * 6), torch.ones_like(amax))
        packed, scales = nvfp4_quantize(
            weight.contiguous(), 1 / scale,
            sfLayout=SfLayout.layout_linear, backend='cute-dsl',
        )
        n, k = weight.shape
        private = private_head_shell(target)
        private.input_size_per_partition = k
        private.output_size_per_partition = n
        private.params_dtype = torch.bfloat16
        private.quant_config = SimpleNamespace(group_size=16)
        private.weight = torch.nn.Parameter(packed.reshape(n, k // 2), requires_grad=False)
        private.weight_scale = torch.nn.Parameter(
            scales.view(torch.float8_e4m3fn).reshape(n, k // 16).contiguous(),
            requires_grad=False,
        )
        private.weight_global_scale = torch.nn.Parameter(scale, requires_grad=False)
        prepare_nvfp4_layer_for_marlin(private)
        private.quant_method = ModelOptNvFp4A16LinearMethod(private.quant_config)
        private.private_draft_nvfp4_a16 = True
        logging.getLogger(__name__).info(
            'PRIVATE_DRAFT_NVFP4_A16 ready shard=%s target_dtype=%s target_ptr=%s private_ptr=%s',
            (n, k), weight.dtype, weight.data_ptr(), private.weight.data_ptr(),
        )
        return private

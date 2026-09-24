#!/usr/bin/env python3
"""Pinned NVFP4 primitives for the RVN PLE converter.

Numerical ground truth is docs/rvn-ple-storage-schema.md §1 together with the
pinned gather reference ``runtime/python/sglang/srt/models/packed_ple.py``
(low nibble first, E2M1 magnitude table, row-major E4M3 block scales per 16
columns, reconstruction ``out = bfloat16(code * scale * g)``).

The multiplier convention is fixed: reconstruction MULTIPLIES by the global
scale ``g`` (never divides); ``g = amax / (6 * 448)`` with the all-zero table
defined to the neutral ``g = 1.0``.
"""
import math
import struct

import torch

# E2M1 magnitude levels for magnitude codes 0..7 (schema §1; packed_ple.py).
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_SIGN = 0x8
GROUP_SIZE = 16
E4M3_MAX = 448.0
ENCODER_VERSION = "rvn-ple-nvfp4-r1"
RECONSTRUCTION = "bf16_direct"
WEIGHT_DTYPE = "e2m1-packed-u8-low-first"
SCALE_DTYPE = "float8_e4m3fn"

_LEVELS = torch.tensor(E2M1_LEVELS, dtype=torch.float32)


def e2m1_decode(nibble: int) -> float:
    """Decode one E2M1 nibble code; 0x8 is IEEE signed zero (-0.0)."""
    if not isinstance(nibble, int) or not 0 <= nibble <= 15:
        raise ValueError(f"E2M1 nibble out of range: {nibble!r}")
    mag = E2M1_LEVELS[nibble & 0x7]
    return -mag if nibble & E2M1_SIGN else mag


def e2m1_encode(value: float) -> int:
    """Encode one finite value to its nearest E2M1 nibble.

    Rounding is round-to-nearest with ties resolved toward the even code (the
    IEEE rule for the E2M1 grid). The sign comes from the IEEE sign bit, so
    -0.0 encodes to 0x8; magnitudes saturate at code 7 (level 6).
    """
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {value!r}")
    sign = E2M1_SIGN if math.copysign(1.0, value) < 0 else 0
    m = min(abs(value), E2M1_LEVELS[-1])
    lo = 0
    for i, level in enumerate(E2M1_LEVELS):
        if level <= m:
            lo = i
    hi = min(lo + 1, 7)
    d_lo, d_hi = m - E2M1_LEVELS[lo], E2M1_LEVELS[hi] - m
    if d_hi < d_lo or (d_hi == d_lo and lo & 1):
        lo = hi
    return sign | lo


def _nearest_codes(mag: torch.Tensor) -> torch.Tensor:
    """Nearest magnitude code 0..7 for ``mag`` in [0, 6]; RNE ties to even code."""
    levels = _LEVELS.to(mag.device)
    lo = torch.searchsorted(levels, mag, right=True).sub_(1).clamp_(0, 7)
    hi = (lo + 1).clamp_(0, 7)
    d_lo = mag - levels[lo]
    d_hi = levels[hi] - mag
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (lo & 1).bool())
    return torch.where(take_hi, hi, lo).to(torch.uint8)


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack odd-width nibble codes ``[..., cols]`` to ``[..., cols//2]`` uint8.

    Low nibble first: value 2i in ``byte[i] & 0x0F``, value 2i+1 in ``byte[i] >> 4``.
    """
    if nibbles.shape[-1] % 2:
        raise ValueError("nibble count must be even")
    lo = nibbles[..., 0::2] & 0x0F
    hi = (nibbles[..., 1::2] & 0x0F) << 4
    return (lo | hi).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles` -> uint8 nibble codes ``[..., cols]``."""
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8,
        device=packed.device)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = packed >> 4
    return out


def e4m3_encode(values: torch.Tensor) -> torch.Tensor:
    """Encode float32 values to ``float8_e4m3fn`` (RNE; finite overflow saturates)."""
    return values.to(torch.float8_e4m3fn)


def e4m3_to_f32(scales: torch.Tensor) -> torch.Tensor:
    """Decode ``float8_e4m3fn`` block scales to float32."""
    return scales.to(torch.float32)


def compute_global_scale(amax: float) -> float:
    """Return ``g = amax / (6 * 448)`` as float32; all-zero table -> neutral 1.0."""
    if not math.isfinite(amax) or amax < 0:
        raise ValueError(f"invalid amax: {amax!r}")
    if amax == 0.0:
        return 1.0
    return struct.unpack("<f", struct.pack("<f", amax / (6.0 * E4M3_MAX)))[0]


def global_scale_bits(g: float) -> int:
    """``struct.pack('<f', g)`` reinterpreted as uint32 (schema §2)."""
    return struct.unpack("<I", struct.pack("<f", g))[0]


def global_scale_from_bits(bits: int) -> float:
    """Inverse of :func:`global_scale_bits`; rejects non-finite/non-positive g."""
    g = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(g) or g <= 0:
        raise ValueError(f"invalid global scale bits: {bits!r}")
    return g


def encode_chunk(x: torch.Tensor, g: float):
    """Encode a float32 ``[rows, cols]`` chunk to (packed uint8, float8 scales).

    Block scale is ``float8_e4m3fn(amax_block / (6 * g))`` per 16 consecutive
    columns (row-major); codes quantize against the *stored* post-rounding
    scale, so ``code * scale * g`` reconstructs the value. A block whose
    pre-scale rounds to zero decodes back to zero either way; its codes are
    forced to plain (unsigned) zero for determinism.
    """
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] % GROUP_SIZE:
        raise ValueError("encode_chunk expects float32 [rows, cols] with cols % 16 == 0")
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    amax_b = blocks.abs().amax(dim=-1)
    scales = e4m3_encode(amax_b / (6.0 * g))
    s_exp = e4m3_to_f32(scales).repeat_interleave(GROUP_SIZE, dim=1)
    safe = s_exp * g
    q = torch.where(safe == 0, torch.zeros_like(x), x / safe)
    codes = _nearest_codes(q.abs().clamp_(max=E2M1_LEVELS[-1]))
    sign = torch.where(x.signbit(), torch.full_like(codes, E2M1_SIGN), 0)
    packed = pack_nibbles(codes | sign)
    return packed, scales.contiguous()


def reconstruct(packed: torch.Tensor, scales: torch.Tensor, g: float) -> torch.Tensor:
    """Decode packed+scales to bfloat16: ``bfloat16(code * scale * g)``, one rounding."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("reconstruct expects uint8 [rows, cols//2]")
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 2:
        raise ValueError("reconstruct expects float8_e4m3fn [rows, cols//16]")
    rows, cols = packed.shape[0], packed.shape[1] * 2
    if scales.shape != (rows, cols // GROUP_SIZE):
        raise ValueError("packed/scales shape mismatch")
    nib = unpack_nibbles(packed)
    levels = _LEVELS.to(packed.device)
    mag = levels[(nib & 0x7).long()]
    val = torch.where((nib & E2M1_SIGN).bool(), -mag, mag)
    s_exp = e4m3_to_f32(scales).repeat_interleave(GROUP_SIZE, dim=1)
    return (val * s_exp * g).to(torch.bfloat16)

"""Reference packers and NumPy math for the SlangML LLM weight/activation formats.

These functions define the on-GPU layouts (docs/quant.md). The future model
loader should repack checkpoints with the same rules.

Weight W is logically [N][K] (out x in); quantization runs along K in blocks of
32 elements. Metadata is one fp16 pair (d, m) per group of `group` elements
(group = 32 * 2^groupShift), dequantized as w = d * q + m.
"""

from __future__ import annotations

import numpy as np

QBLOCK = 32


def f16_bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float16).view(np.uint16)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 with round-to-nearest-even (NaN not handled)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def pack_half2(d: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (f16_bits(d).astype(np.uint32) | (f16_bits(m).astype(np.uint32) << 16)).astype(np.uint32)


def group_shift(group: int) -> int:
    assert group % QBLOCK == 0 and (group // QBLOCK) & (group // QBLOCK - 1) == 0
    return int(np.log2(group // QBLOCK))


def pack_q4(w: np.ndarray, group: int = 32):
    """Affine 4-bit. Returns (codes uint8 [N][K/2], meta uint32 [N][K/group], dequant fp32 [N][K])."""
    n, k = w.shape
    assert k % group == 0
    g = w.reshape(n, k // group, group).astype(np.float32)
    lo = g.min(axis=2)
    hi = g.max(axis=2)
    d = ((hi - lo) / 15.0).astype(np.float16).astype(np.float32)
    m = lo.astype(np.float16).astype(np.float32)
    safe = np.where(d > 0, d, 1.0)
    q = np.clip(np.rint((g - m[..., None]) / safe[..., None]), 0, 15).astype(np.uint8)
    deq = (q.astype(np.float32) * d[..., None] + m[..., None]).reshape(n, k)
    q = q.reshape(n, k // QBLOCK, QBLOCK)
    codes = (q[:, :, :16] | (q[:, :, 16:] << 4)).astype(np.uint8).reshape(n, k // 2)
    return codes, pack_half2(d, m), deq


def pack_q8(w: np.ndarray, group: int = 32):
    """Symmetric int8 (m = 0). Returns (codes int8 [N][K], meta uint32 [N][K/group], dequant)."""
    n, k = w.shape
    g = w.reshape(n, k // group, group).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    safe = np.where(d > 0, d, 1.0)
    q = np.clip(np.rint(g / safe[..., None]), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * d[..., None]).reshape(n, k)
    return q.reshape(n, k), pack_half2(d, np.zeros_like(d)), deq


# OCP MXFP4 (gpt-oss experts). e2m1 code -> value; codes 8..15 are the negatives.
FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def quantize_mxfp4_checkpoint(w: np.ndarray):
    """fp32 [N][K] -> gpt-oss checkpoint layout: blocks uint8 [N][K/32][16], scales uint8 [N][K/32].

    Checkpoint byte j of a block holds element 2j (low nibble) and 2j+1 (high nibble);
    the scale byte is the e8m0 exponent E (value 2^(E-127)). Round to nearest FP4 value.
    """
    n, k = w.shape
    b = w.reshape(n, k // QBLOCK, QBLOCK).astype(np.float64)
    amax = np.abs(b).max(axis=2)
    e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-38) / 6.0)), -127).astype(np.int64)
    e = np.clip(e, -127, 127)
    scaled = b / np.exp2(e)[..., None]
    mags = FP4_VALUES[:8].astype(np.float64)
    idx = np.abs(np.abs(scaled)[..., None] - mags).argmin(axis=-1)
    codes = (idx | np.where(scaled < 0, 8, 0)).astype(np.uint8)  # [n][nb][32]
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    return blocks, (e + 127).astype(np.uint8)


def dequant_mxfp4_checkpoint(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Official gpt_oss/torch/weights.py rule: lut[nibble] * 2^(E-127). Returns fp32 [N][K]."""
    lo = FP4_VALUES[blocks & 0x0F]
    hi = FP4_VALUES[blocks >> 4]
    vals = np.stack([lo, hi], axis=-1).reshape(*blocks.shape[:-1], QBLOCK)
    vals = vals * np.exp2(scales.astype(np.float32) - 127.0)[..., None]
    return vals.reshape(blocks.shape[0], -1).astype(np.float32)


def repack_mxfp4(blocks: np.ndarray, scales: np.ndarray):
    """Checkpoint layout -> SlangML MXFP4: codes uint8 [N][K/2] in split-nibble order
    (byte j = element j low, element j + 16 high), meta = scales uint8 [N][K/32]."""
    n, nb, _ = blocks.shape
    codes = np.stack([blocks & 0x0F, blocks >> 4], axis=-1).reshape(n, nb, QBLOCK)
    split = (codes[..., :16] | (codes[..., 16:] << 4)).astype(np.uint8)
    return split.reshape(n, nb * 16), np.ascontiguousarray(scales.reshape(n, nb))


def pack_mxfp4(w: np.ndarray):
    """Returns (codes uint8 [N][K/2], meta uint8 [N][K/32], dequant fp32 [N][K])."""
    blocks, scales = quantize_mxfp4_checkpoint(w)
    codes, meta = repack_mxfp4(blocks, scales)
    return codes, meta, dequant_mxfp4_checkpoint(blocks, scales)


def pack_weight(fmt: str, w: np.ndarray, group: int = 32):
    """Returns (codes_bytes ndarray, meta ndarray|None, dequantized fp32)."""
    if fmt == "q4":
        return pack_q4(w, group)
    if fmt == "q8":
        return pack_q8(w, group)
    if fmt == "mxfp4":
        return pack_mxfp4(w)
    if fmt == "f16":
        h = w.astype(np.float16)
        return h, None, h.astype(np.float32)
    if fmt == "bf16":
        b = bf16_bits(w)
        return b, None, bf16_to_f32(b)
    raise ValueError(fmt)


def quantize_act_ref(x: np.ndarray):
    """Activation int8 blocks: returns (q int8 [M][K], d [M][K/32], s [M][K/32]); s = d * sum(q)."""
    m, k = x.shape
    b = x.reshape(m, k // QBLOCK, QBLOCK).astype(np.float32)
    amax = np.abs(b).max(axis=2)
    d = amax / 127.0
    inv = np.where(amax > 0, 127.0 / np.where(amax > 0, amax, 1.0), 0.0)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int8)
    return q.reshape(m, k), d, d * q.astype(np.float64).sum(axis=2)


def gemv_q_ref(fmt: str, codes, meta, xq, xs, n: int, k: int, group: int) -> np.ndarray:
    """Exact reference of the int8-dot GEMV given the GPU's quantized activations.

    xq: int8 [M][K]; xs: float32 [M][K/32][2] (d, s).
    """
    nb = k // QBLOCK
    if fmt == "mxfp4":
        c = codes.reshape(n, nb, 16)
        q = np.concatenate([c & 0xF, c >> 4], axis=2)
        q2 = (2.0 * FP4_VALUES[q]).astype(np.int64)  # doubled e2m1 values are integers
        dw = np.exp2(meta.reshape(n, nb).astype(np.float64) - 128.0)
        xqb = xq.reshape(xq.shape[0], nb, QBLOCK).astype(np.int64)
        dot = np.einsum("nbk,mbk->mnb", q2, xqb).astype(np.float64)
        return (dw[None] * xs[:, :, 0].astype(np.float64)[:, None] * dot).sum(axis=2)
    if fmt == "q4":
        c = codes.reshape(n, nb, 16)
        q = np.concatenate([c & 0xF, c >> 4], axis=2).astype(np.int64)  # [n][nb][32]
    else:
        q = codes.reshape(n, nb, QBLOCK).astype(np.int64)
    meta = meta.reshape(n, k // group)
    d = meta.astype(np.uint16).view(np.float16).astype(np.float64)
    mm = (meta >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    gidx = np.arange(nb) // (group // QBLOCK)
    d, mm = d[:, gidx], mm[:, gidx]  # [n][nb]
    xqb = xq.reshape(xq.shape[0], nb, QBLOCK).astype(np.int64)
    dot = np.einsum("nbk,mbk->mnb", q, xqb).astype(np.float64)  # [m][n][nb]
    dx = xs[:, :, 0].astype(np.float64)
    sx = xs[:, :, 1].astype(np.float64)
    return (d[None] * dx[:, None] * dot + mm[None] * sx[:, None]).sum(axis=2)

// SlangML LLM helper module: weight-format tags, packed-format decode, int8
// dot products, activation helpers and small subgroup utilities used by the
// quant/, llm/ and linear_attention/ families.
//
// Import with:
//     import slangml_common;
//     import slangml_llm;
// See docs/quant.md for the bit layouts and docs/LLM_ROADMAP.md for status.
module slangml_llm;

import slangml_common;

// ---------------------------------------------------------------------------
// Weight formats. Every quantized weight is logically [N][K] (out x in) and is
// quantized along K in blocks of 32. Codes and per-group metadata are separate
// arrays. Metadata is one packed fp16 pair (d, m) per group: w = d * q + m.
// ---------------------------------------------------------------------------
public static const uint SML_WFMT_F32 = 0u;
public static const uint SML_WFMT_F16 = 1u;
public static const uint SML_WFMT_BF16 = 2u;
public static const uint SML_WFMT_Q8 = 3u;  // int8 codes, 32 bytes per block
public static const uint SML_WFMT_Q4 = 4u;  // uint4 codes, 16 bytes per block (split-nibble order)
// OCP MXFP4 (gpt-oss experts): e2m1 codes, 16 bytes per block in split-nibble
// order (like Q4), plus one e8m0 exponent byte per block (group is always 32).
public static const uint SML_WFMT_MXFP4 = 5u;

public static const uint SML_QBLOCK = 32u;
public static const uint SML_Q4_BLOCK_BYTES = 16u;
public static const uint SML_Q8_BLOCK_BYTES = 32u;

// Base byte address of a tensor whose element size is `elementBytes`.
// The descriptor's baseOffset is in elements (contract section 3.2).
[ForceInline]
public uint64_t sml_base(TensorDesc t, uint elementBytes)
{
    return t.dataAddress + uint64_t(t.baseOffset) * uint64_t(elementBytes);
}

// ---------------------------------------------------------------------------
// Aligned BDA access. With -fvk-use-scalar-layout, plain `p[i]` on a float4 /
// uint4 pointer is emitted as `Aligned 4`, which forces the driver to split
// every 128-bit access into four 32-bit ones (measured 2.6x slower prefill
// GEMM on GA107). These helpers state the real alignment. Callers guarantee
// it: LLM tensors are 16-byte aligned and their rows are multiples of 32 elements.
// ---------------------------------------------------------------------------
[ForceInline] public float4 sml_ld_f4(float4* p) { return loadAligned<16>(p); }
[ForceInline] public uint4 sml_ld_u4(uint4* p) { return loadAligned<16>(p); }
[ForceInline] public uint2 sml_ld_u2(uint2* p) { return loadAligned<8>(p); }
[ForceInline] public float2 sml_ld_f2(float2* p) { return loadAligned<8>(p); }
[ForceInline] public void sml_st_f4(float4* p, float4 v) { storeAligned<16>(p, v); }
[ForceInline] public void sml_st_u4(uint4* p, uint4 v) { storeAligned<16>(p, v); }
[ForceInline] public void sml_st_f2(float2* p, float2 v) { storeAligned<8>(p, v); }

// ---------------------------------------------------------------------------
// Packed float decode
// ---------------------------------------------------------------------------
[ForceInline]
public float2 sml_half2_to_float2(uint bits)
{
    return float2(f16tof32(bits), f16tof32(bits >> 16u));
}

[ForceInline]
public float2 sml_bf16x2_to_float2(uint bits)
{
    return float2(asfloat(bits << 16u), asfloat(bits & 0xFFFF0000u));
}

[ForceInline]
public uint sml_float2_to_half2(float2 v)
{
    return (f32tof16(v.x) & 0xFFFFu) | (f32tof16(v.y) << 16u);
}

// Eight fp16 values packed in a uint4 -> two float4 (elements 0..3, 4..7).
[ForceInline]
public void sml_unpack_half8(uint4 w, out float4 lo, out float4 hi)
{
    float2 a = sml_half2_to_float2(w.x);
    float2 b = sml_half2_to_float2(w.y);
    float2 c = sml_half2_to_float2(w.z);
    float2 d = sml_half2_to_float2(w.w);
    lo = float4(a, b);
    hi = float4(c, d);
}

[ForceInline]
public void sml_unpack_bf16x8(uint4 w, out float4 lo, out float4 hi)
{
    float2 a = sml_bf16x2_to_float2(w.x);
    float2 b = sml_bf16x2_to_float2(w.y);
    float2 c = sml_bf16x2_to_float2(w.z);
    float2 d = sml_bf16x2_to_float2(w.w);
    lo = float4(a, b);
    hi = float4(c, d);
}

// ---------------------------------------------------------------------------
// int8 dot products (lower to OpSDot with PackedVectorFormat4x8Bit)
// ---------------------------------------------------------------------------
[ForceInline]
public int sml_dp4a(uint a, uint b, int acc)
{
    return dot4add_i8packed(a, b, acc);
}

// Q4 block (16 bytes) times an int8 activation block (two uint4 = 32 bytes).
// Split-nibble order: byte j holds element j (low nibble) and j+16 (high
// nibble), so `w & 0x0F0F0F0F` lines up with activation bytes 0..15 and
// `(w >> 4) & 0x0F0F0F0F` with bytes 16..31. Nibbles are 0..15, which are
// non-negative as signed int8, so a signed dot is exact.
[ForceInline]
public int sml_q4_dot(uint4 w, uint4 x0, uint4 x1)
{
    const uint mask = 0x0F0F0F0Fu;
    int acc = 0;
    acc = sml_dp4a(w.x & mask, x0.x, acc);
    acc = sml_dp4a(w.y & mask, x0.y, acc);
    acc = sml_dp4a(w.z & mask, x0.z, acc);
    acc = sml_dp4a(w.w & mask, x0.w, acc);
    acc = sml_dp4a((w.x >> 4u) & mask, x1.x, acc);
    acc = sml_dp4a((w.y >> 4u) & mask, x1.y, acc);
    acc = sml_dp4a((w.z >> 4u) & mask, x1.z, acc);
    acc = sml_dp4a((w.w >> 4u) & mask, x1.w, acc);
    return acc;
}

[ForceInline]
public int sml_q8_dot(uint4 w0, uint4 w1, uint4 x0, uint4 x1)
{
    int acc = 0;
    acc = sml_dp4a(w0.x, x0.x, acc);
    acc = sml_dp4a(w0.y, x0.y, acc);
    acc = sml_dp4a(w0.z, x0.z, acc);
    acc = sml_dp4a(w0.w, x0.w, acc);
    acc = sml_dp4a(w1.x, x1.x, acc);
    acc = sml_dp4a(w1.y, x1.y, acc);
    acc = sml_dp4a(w1.z, x1.z, acc);
    acc = sml_dp4a(w1.w, x1.w, acc);
    return acc;
}

// ---------------------------------------------------------------------------
// MXFP4 (OCP microscaling FP4). Code nibble = s e1 e0 m (e2m1); values
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign. Doubled they are the integers
// {0, 1, 2, 3, 4, 6, 8, 12}, so a block dequantizes as
//   w = 2^(E - 127) * e2m1 = 2^(E - 128) * mag2 * sign
// and the dot product runs on int8 dp4a like Q4.
// ---------------------------------------------------------------------------

// Four e2m1 nibbles, one in the low nibble of each byte of `n` (high nibbles
// zero) -> doubled magnitudes as bytes, split by sign: `pos` holds the bytes of
// positive codes, `neg` those of negative codes (the other bytes are 0).
// SWAR: magnitude codes 0..3 map to themselves; 4 -> 4, 5 -> 6, 6 -> 8, 7 -> 12,
// i.e. mag = m + [m >= 5] + [m >= 6] + 3 [m >= 7]. No byte carries (m + 3 <= 10).
[ForceInline]
public void sml_e2m1x4_decode(uint n, out uint pos, out uint neg)
{
    uint m = n & 0x07070707u;
    uint smask = ((n >> 3u) & 0x01010101u) * 0xFFu;
    uint t5 = (m + 0x03030303u) & 0x08080808u;
    uint t6 = (m + 0x02020202u) & 0x08080808u;
    uint t7 = (m + 0x01010101u) & 0x08080808u;
    uint mag = m + ((t5 + t6 + t7 * 3u) >> 3u);  // per-byte sum is a multiple of 8: no bits cross bytes
    pos = mag & ~smask;
    neg = mag & smask;
}

// MXFP4 block (16 bytes, split-nibble order) times an int8 activation block.
// Returns sum(mag2 * sign * x); multiply by 2^(E-128) * d_x.
[ForceInline]
public int sml_mxfp4_dot(uint4 w, uint4 x0, uint4 x1)
{
    const uint mask = 0x0F0F0F0Fu;
    int accP = 0;
    int accN = 0;
    uint p, n;
    sml_e2m1x4_decode(w.x & mask, p, n); accP = sml_dp4a(p, x0.x, accP); accN = sml_dp4a(n, x0.x, accN);
    sml_e2m1x4_decode(w.y & mask, p, n); accP = sml_dp4a(p, x0.y, accP); accN = sml_dp4a(n, x0.y, accN);
    sml_e2m1x4_decode(w.z & mask, p, n); accP = sml_dp4a(p, x0.z, accP); accN = sml_dp4a(n, x0.z, accN);
    sml_e2m1x4_decode(w.w & mask, p, n); accP = sml_dp4a(p, x0.w, accP); accN = sml_dp4a(n, x0.w, accN);
    sml_e2m1x4_decode((w.x >> 4u) & mask, p, n); accP = sml_dp4a(p, x1.x, accP); accN = sml_dp4a(n, x1.x, accN);
    sml_e2m1x4_decode((w.y >> 4u) & mask, p, n); accP = sml_dp4a(p, x1.y, accP); accN = sml_dp4a(n, x1.y, accN);
    sml_e2m1x4_decode((w.z >> 4u) & mask, p, n); accP = sml_dp4a(p, x1.z, accP); accN = sml_dp4a(n, x1.z, accN);
    sml_e2m1x4_decode((w.w >> 4u) & mask, p, n); accP = sml_dp4a(p, x1.w, accP); accN = sml_dp4a(n, x1.w, accN);
    return accP - accN;
}

// e8m0 exponent byte -> 2^(E - 128) exactly (the scale of the doubled codes).
// E = 255 (NaN in OCP) is not produced by gpt-oss checkpoints and maps to 2^127.
[ForceInline]
public float sml_e8m0_half_scale(uint e)
{
    return e >= 2u ? asfloat((e - 1u) << 23u) : asfloat(0x00200000u << e);
}

// Exponent byte `blockIndex` of a packed uint8 scale array.
[ForceInline]
public uint sml_e8m0_load(uint* scales, uint blockIndex)
{
    return (scales[blockIndex >> 2u] >> ((blockIndex & 3u) * 8u)) & 0xFFu;
}

// Pack four already-rounded int values in [-128, 127] into one uint.
[ForceInline]
public uint sml_pack_i8x4(int4 q)
{
    return (uint(q.x) & 0xFFu) | ((uint(q.y) & 0xFFu) << 8u)
        | ((uint(q.z) & 0xFFu) << 16u) | ((uint(q.w) & 0xFFu) << 24u);
}

[ForceInline]
public int sml_sext8(uint v)
{
    return int(v << 24u) >> 24;
}

// ---------------------------------------------------------------------------
// Dequantize helpers (used by GEMM tile loads and embedding lookup)
// ---------------------------------------------------------------------------

// Element `e` (0..31) of a Q4 block -> code.
[ForceInline]
public uint sml_q4_code(uint4 w, uint e)
{
    uint byteIndex = e & 15u;
    uint word = byteIndex < 8u ? (byteIndex < 4u ? w.x : w.y) : (byteIndex < 12u ? w.z : w.w);
    uint b = (word >> ((byteIndex & 3u) * 8u)) & 0xFFu;
    return e < 16u ? (b & 0xFu) : (b >> 4u);
}

// Word j (0..3) of a Q4 block -> elements 4j..4j+3 (lo) and 16+4j..16+4j+3 (hi).
[ForceInline]
public void sml_q4_word_to_float(uint word, float2 dm, out float4 lo, out float4 hi)
{
    lo = float4(float(word & 0xFu), float((word >> 8u) & 0xFu),
                float((word >> 16u) & 0xFu), float((word >> 24u) & 0xFu)) * dm.x + dm.y;
    hi = float4(float((word >> 4u) & 0xFu), float((word >> 12u) & 0xFu),
                float((word >> 20u) & 0xFu), float(word >> 28u)) * dm.x + dm.y;
}

// Word of a Q8 block -> 4 consecutive elements.
[ForceInline]
public float4 sml_q8_word_to_float(uint word, float2 dm)
{
    return float4(float(sml_sext8(word)), float(sml_sext8(word >> 8u)),
                  float(sml_sext8(word >> 16u)), float(int(word) >> 24)) * dm.x + dm.y;
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------
public static const float SML_LOG2E = 1.4426950408889634f;

[ForceInline]
public float sml_sigmoid(float x)
{
    // exp2 overflows to +inf for very negative x, giving exactly 0.
    return 1.0f / (1.0f + exp2(-x * SML_LOG2E));
}

[ForceInline]
public float sml_silu(float x)
{
    return x * sml_sigmoid(x);
}

// PyTorch softplus (beta = 1, threshold = 20).
[ForceInline]
public float sml_softplus(float x)
{
    return x > 20.0f ? x : log(1.0f + exp(x));
}

[ForceInline]
public float sml_gelu_tanh(float x)
{
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

public static const uint SML_ACT_SWIGLU = 0u;       // silu(g) * u
public static const uint SML_ACT_GEGLU_TANH = 1u;   // gelu_tanh(g) * u
public static const uint SML_ACT_SWIGLU_OAI = 2u;   // gpt-oss clamped SwiGLU

// Gated activation used by the fused gate/up GEMV and the standalone kernel.
[ForceInline]
public float sml_glu(uint mode, float g, float u, float limit, float alpha)
{
    if (mode == SML_ACT_GEGLU_TANH)
        return sml_gelu_tanh(g) * u;
    if (mode == SML_ACT_SWIGLU_OAI)
    {
        g = min(g, limit);
        u = clamp(u, -limit, limit);
        return (u + 1.0f) * g * sml_sigmoid(alpha * g);
    }
    return sml_silu(g) * u;
}

// ---------------------------------------------------------------------------
// Subgroup cluster reductions over aligned groups of 2/4/8/16 lanes.
// All lanes of the cluster must be active.
// ---------------------------------------------------------------------------
[ForceInline]
public float sml_cluster_sum(float v, uint clusterSize)
{
    uint lane = WaveGetLaneIndex();
    for (uint offset = 1u; offset < clusterSize; offset <<= 1u)
        v += WaveReadLaneAt(v, lane ^ offset);
    return v;
}

[ForceInline]
public float sml_cluster_max(float v, uint clusterSize)
{
    uint lane = WaveGetLaneIndex();
    for (uint offset = 1u; offset < clusterSize; offset <<= 1u)
        v = max(v, WaveReadLaneAt(v, lane ^ offset));
    return v;
}

// Order-preserving float -> uint key (larger float -> larger key).
[ForceInline]
public uint sml_float_order_key(float v)
{
    uint bits = asuint(v);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

[ForceInline]
public float sml_float_from_order_key(uint key)
{
    return asfloat((key & 0x80000000u) != 0u ? (key & 0x7FFFFFFFu) : ~key);
}

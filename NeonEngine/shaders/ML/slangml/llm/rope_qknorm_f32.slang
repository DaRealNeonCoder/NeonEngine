// QK-norm + rotary embedding + KV-cache append in one pass (docs/llm_ops.md).
//
// For each token t and head:
//   Q heads:  q = rope(norm_q(Qsrc[t][h]))          -> Qout fp32 [T][Hq][D]
//   K heads:  k = rope(norm_k(Ksrc[t][h]))          -> Kcache fp16 [Hkv][maxCtx][D] at slot(t)
//   V heads:  v = Vsrc[t][h]                         -> Vcache fp16 [Hkv][maxCtx][D] at slot(t)
//
// norm: RMSNorm over D (fp32), weight w or (1 + w) (SML_ROPE_UNIT_OFFSET).
// rope: the first `rotaryDim` dims are rotated, the rest pass through.
//   NeoX / rotate-half (default): pairs (i, i + rd/2) share frequency i.
//   GPT-J interleaved (SML_ROPE_INTERLEAVED): pairs (2i, 2i + 1).
//   inv_freq[i] = theta^(-2i/rd), or a host table (SML_ROPE_FREQ_TABLE) for
//   linear / NTK / YaRN / llama3 scaling; cos and sin are multiplied by
//   `attnFactor` (YaRN mscale; 1 otherwise).
//   MRoPE (SML_ROPE_MROPE): positions are [3][posStride] (T, H, W ids) and
//   frequency i takes its id from a section: interleaved (Qwen3-VL/3.5,
//   SML_ROPE_MROPE_INTERLEAVED): H if i%3==1 && i<3*s1, W if i%3==2 && i<3*s2,
//   else T; contiguous (Qwen2-VL): T for i<s0, H for i<s0+s1, else W.
//   The angle is computed as float(pos) * inv_freq in fp32 (as in HF), then
//   range-reduced so sin/cos stay accurate at 256K-token positions.
// slot(t) = slots[t] with SML_ROPE_SLOT_TABLE, else slotBase + t.
//
// Grid: x = Hq + Hkv (+ Hkv when SML_ROPE_WRITE_V), y = T. 64 threads; each
// thread owns dims d = tid + 64 j. D must be even and <= 64 * SML_ROPE_MAX_DPT.
//
// Tensor table: 0 Qsrc, 1 Ksrc, 2 Vsrc, 3 qNorm w, 4 kNorm w, 5 positions i32,
//               6 Qout, 7 Kcache, 8 Vcache, 9 inv_freq table, 10 slots i32.
import slangml_common;
import slangml_llm;

#ifndef SML_ROPE_MAX_DPT
#define SML_ROPE_MAX_DPT 4
#endif

static const uint SML_ROPE_NORM_Q = 1u;
static const uint SML_ROPE_NORM_K = 2u;
static const uint SML_ROPE_UNIT_OFFSET = 4u;
static const uint SML_ROPE_INTERLEAVED = 8u;
static const uint SML_ROPE_MROPE = 16u;
static const uint SML_ROPE_MROPE_INTERLEAVED = 32u;
static const uint SML_ROPE_FREQ_TABLE = 64u;
static const uint SML_ROPE_WRITE_V = 128u;
static const uint SML_ROPE_SLOT_TABLE = 256u;

struct RopeParams
{
    KernelHeader header;
    uint T;               // 48
    uint numQHeads;       // 52
    uint numKVHeads;      // 56
    uint headDim;         // 60
    uint rotaryDim;       // 64
    uint flags;           // 68
    float eps;            // 72
    float log2Theta;      // 76
    float attnFactor;     // 80
    uint qTokenStride;    // 84 element strides of the source projections
    uint qHeadStride;     // 88
    uint kTokenStride;    // 92
    uint kHeadStride;     // 96
    uint vTokenStride;    // 100
    uint vHeadStride;     // 104
    uint maxCtx;          // 108
    uint slotBase;        // 112
    uint posStride;       // 116 MRoPE: elements between the T/H/W id rows
    uint section0;        // 120
    uint section1;        // 124
    uint section2;        // 128
    uint reserved;        // 132
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlRopePush;

groupshared float smlRopeVec[64 * SML_ROPE_MAX_DPT];
groupshared float smlRopeSum[2];

static const float SML_TWO_PI_HI = 6.28318548202514648f;      // fp32(2*pi)
static const float SML_TWO_PI_LO = -1.7484556e-07f;           // 2*pi - fp32(2*pi)
static const float SML_INV_TWO_PI = 0.15915494309189535f;

[ForceInline]
void sml_sincos_reduced(float angle, out float s, out float c)
{
    float k = round(angle * SML_INV_TWO_PI);
    float r = fma(-k, SML_TWO_PI_HI, angle);
    r = fma(-k, SML_TWO_PI_LO, r);
    s = sin(r);
    c = cos(r);
}

[ForceInline]
uint sml_rope_section(RopeParams* p, uint i)
{
    if ((p.flags & SML_ROPE_MROPE_INTERLEAVED) != 0u)
    {
        uint m3 = i % 3u;
        if (m3 == 1u && i < 3u * p.section1)
            return 1u;
        if (m3 == 2u && i < 3u * p.section2)
            return 2u;
        return 0u;
    }
    if (i < p.section0)
        return 0u;
    if (i < p.section0 + p.section1)
        return 1u;
    return 2u;
}

// Stores this thread's dims d = tid + 64 j as fp16. Neighbouring lanes pair up
// (d, d + 1) so every store is a packed 32-bit word; the even lane writes.
// (Also avoids 16-bit BDA stores, which the AMD iGPU driver mishandles.)
[ForceInline]
void sml_rope_store_f16(uint64_t rowAddress, float v[SML_ROPE_MAX_DPT], uint tidx, uint D)
{
    uint* row = reinterpret<uint*>(rowAddress);
    [ForceUnroll]
    for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
    {
        uint d = tidx + 64u * j;
        float next = WaveReadLaneAt(v[j], WaveGetLaneIndex() ^ 1u);
        if ((tidx & 1u) == 0u && d < D)
            row[d >> 1u] = sml_float2_to_half2(float2(v[j], next));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_llm_rope_qknorm_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RopeParams* p = reinterpret<RopeParams*>(smlRopePush.headerAddress);
    uint t = p.header.dispatchBase.y + gid.y;
    uint head = p.header.dispatchBase.x + gid.x;
    if (t >= p.T)
        return;  // uniform per workgroup
    uint D = p.headDim;
    uint Hq = p.numQHeads;
    uint Hkv = p.numKVHeads;
    // role: 0 = Q, 1 = K, 2 = V
    uint role = head < Hq ? 0u : (head < Hq + Hkv ? 1u : 2u);
    uint h = role == 0u ? head : (role == 1u ? head - Hq : head - Hq - Hkv);
    if (role == 2u && (h >= Hkv || (p.flags & SML_ROPE_WRITE_V) == 0u))
        return;

    uint slot = p.slotBase + t;
    if ((p.flags & SML_ROPE_SLOT_TABLE) != 0u)
        slot = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 10u), 4u))[t];

    uint srcIndex = role == 0u ? 0u : (role == 1u ? 1u : 2u);
    uint tokenStride = role == 0u ? p.qTokenStride : (role == 1u ? p.kTokenStride : p.vTokenStride);
    uint headStride = role == 0u ? p.qHeadStride : (role == 1u ? p.kHeadStride : p.vHeadStride);
    float* src = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, srcIndex), 4u))
        + (t * tokenStride + h * headStride);

    float x[SML_ROPE_MAX_DPT];
    float sq = 0.0f;
    [ForceUnroll]
    for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
    {
        uint d = tid.x + 64u * j;
        x[j] = d < D ? src[d] : 0.0f;
        sq += x[j] * x[j];
    }

    if (role == 2u)
    {
        sml_rope_store_f16(sml_base(slangml_tensor_desc(p.header, 8u), 2u)
                           + (uint64_t(h) * p.maxCtx + slot) * D * 2u, x, tid.x, D);
        return;
    }

    bool doNorm = (p.flags & (role == 0u ? SML_ROPE_NORM_Q : SML_ROPE_NORM_K)) != 0u;
    if (doNorm)
    {
        // 64 threads = one or two subgroups.
        float s = WaveActiveSum(sq);
        uint laneCount = WaveGetLaneCount();
        if (WaveGetLaneIndex() == 0u)
            smlRopeSum[tid.x / laneCount] = s;
        GroupMemoryBarrierWithGroupSync();
        float total = laneCount >= 64u ? smlRopeSum[0] : smlRopeSum[0] + smlRopeSum[1];
        float inv = rsqrt(total / float(D) + p.eps);
        float* w = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, role == 0u ? 3u : 4u), 4u));
        float wOffset = (p.flags & SML_ROPE_UNIT_OFFSET) != 0u ? 1.0f : 0.0f;
        [ForceUnroll]
        for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
        {
            uint d = tid.x + 64u * j;
            if (d < D)
                x[j] = x[j] * inv * (w[d] + wOffset);
        }
    }

    [ForceUnroll]
    for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
        smlRopeVec[tid.x + 64u * j] = x[j];
    GroupMemoryBarrierWithGroupSync();

    uint rd = p.rotaryDim;
    uint half = rd >> 1u;
    bool interleaved = (p.flags & SML_ROPE_INTERLEAVED) != 0u;
    int* pos = reinterpret<int*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float* freqTable = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 9u), 4u));
    float out[SML_ROPE_MAX_DPT];
    [ForceUnroll]
    for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
    {
        uint d = tid.x + 64u * j;
        out[j] = x[j];
        if (d < rd)
        {
            uint i = interleaved ? (d >> 1u) : (d < half ? d : d - half);
            uint partner = interleaved ? (d ^ 1u) : (d < half ? d + half : d - half);
            bool first = interleaved ? ((d & 1u) == 0u) : (d < half);
            uint sec = (p.flags & SML_ROPE_MROPE) != 0u ? sml_rope_section(p, i) : 0u;
            float position = float(pos[sec * p.posStride + t]);
            float invFreq = (p.flags & SML_ROPE_FREQ_TABLE) != 0u
                ? freqTable[i]
                : exp2(-float(2u * i) / float(rd) * p.log2Theta);
            float s, c;
            sml_sincos_reduced(position * invFreq, s, c);
            s *= p.attnFactor;
            c *= p.attnFactor;
            float other = smlRopeVec[partner];
            out[j] = first ? x[j] * c - other * s : x[j] * c + other * s;
        }
    }

    if (role == 0u)
    {
        float* qo = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u)) + (t * Hq + h) * D;
        [ForceUnroll]
        for (uint j = 0; j < SML_ROPE_MAX_DPT; ++j)
        {
            uint d = tid.x + 64u * j;
            if (d < D)
                qo[d] = out[j];
        }
    }
    else
    {
        sml_rope_store_f16(sml_base(slangml_tensor_desc(p.header, 7u), 2u)
                           + (uint64_t(h) * p.maxCtx + slot) * D * 2u, out, tid.x, D);
    }
}

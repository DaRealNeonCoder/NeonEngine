// Gated DeltaNet recurrence (Qwen3-Next / Qwen3.5 / Qwen3.8 linear attention),
// decode step and prefill scan in one kernel (docs/linear_attention.md).
//
// Per V head h (K head kh = h / (Hv / Hk), i.e. HF repeat_interleave) and token t:
//   q = l2norm(q_t[kh]) / sqrt(dk),  k = l2norm(k_t[kh])      l2norm: x * rsqrt(sum x^2 + eps)
//   g = -exp(A_log[h]) * softplus(a_t[h] + dt_bias[h]),  beta = sigmoid(b_t[h])
//   S = exp(g) * S                                  S is fp32 [dk][dv], kept in registers
//   delta = beta * (v_t - S^T k)
//   S = S + k delta^T
//   o_t = S^T q
// SML_GDN_GATES_PRECOMPUTED: a holds g and b holds beta directly.
//
// Mapping: one 128-thread workgroup per (V head, block of 32 value columns).
// Within each 32-lane group, lane = col8 + 8 * slice: 8 columns x 4 row slices
// of dk/4 rows, so every thread holds dk/4 state values and the dk reduction
// is two xor-shuffles (no barriers per token). Tokens are staged 16 at a time
// in shared memory (q, k, v, norms, gates) so global loads overlap.
// Grid: x = dv / 32, y = Hv.  dk = SML_GDN_DK (default 128), dv multiple of 32.
//
// Tensor table: 0 qkv [T][ldQKV] (q at qOffset, k at kOffset, v at vOffset,
//   heads contiguous), 1 a [T][ldAB], 2 b [T][ldAB], 3 A_log [Hv],
//   4 dt_bias [Hv], 5 state fp32 [Hv][dk][dv] (in/out), 6 out [T][ldOut].
import slangml_common;
import slangml_llm;

#ifndef SML_GDN_DK
#define SML_GDN_DK 128
#endif

static const uint SML_GDN_ROWS = SML_GDN_DK / 4u;  // state rows per thread
static const uint SML_GDN_TCH = 16u;               // tokens staged per step
static const uint SML_GDN_GATES_PRECOMPUTED = 1u;

struct GdnParams
{
    KernelHeader header;
    uint T;          // 48
    uint numKHeads;  // 52
    uint numVHeads;  // 56
    uint dv;         // 60
    uint ldQKV;      // 64
    uint qOffset;    // 68
    uint kOffset;    // 72
    uint vOffset;    // 76
    uint ldAB;       // 80
    uint ldOut;      // 84
    float qScale;    // 88  usually 1 / sqrt(dk)
    float eps;       // 92  l2norm epsilon (1e-6)
    uint flags;      // 96
    uint reserved;   // 100
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGdnPush;

groupshared float4 smlGdnQ[SML_GDN_TCH * SML_GDN_DK / 4];
groupshared float4 smlGdnK[SML_GDN_TCH * SML_GDN_DK / 4];
groupshared float smlGdnV[SML_GDN_TCH * 32];
groupshared float4 smlGdnTok[SML_GDN_TCH];  // (q scale, k scale, k.q after scaling, unused)
groupshared float2 smlGdnGate[SML_GDN_TCH]; // (decay, beta)

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_linattn_gated_delta_rule_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    GdnParams* p = reinterpret<GdnParams*>(smlGdnPush.headerAddress);
    uint h = gid.y;
    uint colBlock = gid.x;
    uint lane = tid.x & 31u;
    uint group = tid.x >> 5u;
    uint col8 = lane & 7u;
    uint slice = lane >> 3u;
    uint col = colBlock * 32u + group * 8u + col8;  // value column owned by this thread
    uint row0 = slice * SML_GDN_ROWS;
    uint kh = h / (p.numVHeads / p.numKHeads);

    uint64_t qkvBase = sml_base(slangml_tensor_desc(p.header, 0u), 4u);
    float* qkv = reinterpret<float*>(qkvBase);
    float* aIn = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u));
    float* bIn = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u));
    float aLog = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u))[h];
    float dtBias = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u))[h];
    // Byte-address math: `pointer + uint64` (64-bit OpPtrAccessChain index) is
    // miscompiled by the AMD iGPU driver.
    float* state = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u)
                                       + uint64_t(h) * SML_GDN_DK * p.dv * 4u);
    float* out = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u));
    bool precomputed = (p.flags & SML_GDN_GATES_PRECOMPUTED) != 0u;
    float negA = -exp(aLog);

    float S[SML_GDN_ROWS];
    [ForceUnroll]
    for (uint i = 0; i < SML_GDN_ROWS; ++i)
        S[i] = state[(row0 + i) * p.dv + col];

    for (uint t0 = 0; t0 < p.T; t0 += SML_GDN_TCH)
    {
        uint nt = min(SML_GDN_TCH, p.T - t0);
        GroupMemoryBarrierWithGroupSync();  // previous chunk's readers are done
        // Stage q, k (dk floats each per token) and this block's 32 v columns.
        for (uint idx = tid.x; idx < SML_GDN_TCH * SML_GDN_DK / 4u; idx += 128u)
        {
            uint tt = idx / (SML_GDN_DK / 4u);
            uint d4 = idx - tt * (SML_GDN_DK / 4u);
            float4 qv = float4(0.0f);
            float4 kv = float4(0.0f);
            if (tt < nt)
            {
                // Integer address math: casting a pointer that already carries an
                // offset (float* + n -> float4*) is miscompiled by the AMD iGPU driver.
                uint64_t rowAddr = qkvBase + uint64_t((t0 + tt) * p.ldQKV) * 4u;
                qv = sml_ld_f4(reinterpret<float4*>(rowAddr + uint64_t(p.qOffset + kh * SML_GDN_DK + d4 * 4u) * 4u));
                kv = sml_ld_f4(reinterpret<float4*>(rowAddr + uint64_t(p.kOffset + kh * SML_GDN_DK + d4 * 4u) * 4u));
            }
            smlGdnQ[idx] = qv;
            smlGdnK[idx] = kv;
        }
        for (uint idx = tid.x; idx < SML_GDN_TCH * 32u; idx += 128u)
        {
            uint tt = idx >> 5u;
            uint c = idx & 31u;
            smlGdnV[idx] = tt < nt ? qkv[(t0 + tt) * p.ldQKV + p.vOffset + h * p.dv + colBlock * 32u + c] : 0.0f;
        }
        if (tid.x < SML_GDN_TCH)
        {
            float2 gate = float2(1.0f, 0.0f);
            if (tid.x < nt)
            {
                float a = aIn[(t0 + tid.x) * p.ldAB + h];
                float b = bIn[(t0 + tid.x) * p.ldAB + h];
                float g = precomputed ? a : negA * sml_softplus(a + dtBias);
                gate = float2(exp(g), precomputed ? b : sml_sigmoid(b));
            }
            smlGdnGate[tid.x] = gate;
        }
        GroupMemoryBarrierWithGroupSync();
        // Per-token norms and k.q: 8 lanes per token, each summing dk/8 values.
        {
            uint tt = tid.x >> 3u;
            uint part = tid.x & 7u;
            float qq = 0.0f, kk = 0.0f, kq = 0.0f;
            [ForceUnroll]
            for (uint j = 0; j < SML_GDN_DK / 32u; ++j)
            {
                float4 qv = smlGdnQ[tt * (SML_GDN_DK / 4u) + part * (SML_GDN_DK / 32u) + j];
                float4 kv = smlGdnK[tt * (SML_GDN_DK / 4u) + part * (SML_GDN_DK / 32u) + j];
                qq += dot(qv, qv);
                kk += dot(kv, kv);
                kq += dot(kv, qv);
            }
            qq = sml_cluster_sum(qq, 8u);
            kk = sml_cluster_sum(kk, 8u);
            kq = sml_cluster_sum(kq, 8u);
            if (part == 0u)
            {
                float qs = rsqrt(qq + p.eps) * p.qScale;
                float ks = rsqrt(kk + p.eps);
                smlGdnTok[tt] = float4(qs, ks, kq * qs * ks, 0.0f);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint tt = 0; tt < nt; ++tt)
        {
            float4 tok = smlGdnTok[tt];
            float2 gate = smlGdnGate[tt];
            float sk = 0.0f, sq = 0.0f;
            [ForceUnroll]
            for (uint i4 = 0; i4 < SML_GDN_ROWS / 4u; ++i4)
            {
                float4 kv = smlGdnK[tt * (SML_GDN_DK / 4u) + row0 / 4u + i4];
                float4 qv = smlGdnQ[tt * (SML_GDN_DK / 4u) + row0 / 4u + i4];
                sk += S[i4 * 4u + 0u] * kv.x + S[i4 * 4u + 1u] * kv.y + S[i4 * 4u + 2u] * kv.z + S[i4 * 4u + 3u] * kv.w;
                sq += S[i4 * 4u + 0u] * qv.x + S[i4 * 4u + 1u] * qv.y + S[i4 * 4u + 2u] * qv.z + S[i4 * 4u + 3u] * qv.w;
            }
            // Reduce over the four row slices (lanes xor 8, xor 16).
            sk += WaveReadLaneAt(sk, WaveGetLaneIndex() ^ 8u);
            sq += WaveReadLaneAt(sq, WaveGetLaneIndex() ^ 8u);
            sk += WaveReadLaneAt(sk, WaveGetLaneIndex() ^ 16u);
            sq += WaveReadLaneAt(sq, WaveGetLaneIndex() ^ 16u);
            sk *= tok.y;  // apply k normalisation
            sq *= tok.x;  // apply q normalisation and scale
            float decay = gate.x;
            float vcol = smlGdnV[tt * 32u + group * 8u + col8];
            float delta = gate.y * (vcol - decay * sk);
            float o = decay * sq + delta * tok.z;
            float dk = delta * tok.y;  // k_i * delta = raw k_i * (ks * delta)
            [ForceUnroll]
            for (uint i4 = 0; i4 < SML_GDN_ROWS / 4u; ++i4)
            {
                float4 kv = smlGdnK[tt * (SML_GDN_DK / 4u) + row0 / 4u + i4];
                S[i4 * 4u + 0u] = fma(decay, S[i4 * 4u + 0u], kv.x * dk);
                S[i4 * 4u + 1u] = fma(decay, S[i4 * 4u + 1u], kv.y * dk);
                S[i4 * 4u + 2u] = fma(decay, S[i4 * 4u + 2u], kv.z * dk);
                S[i4 * 4u + 3u] = fma(decay, S[i4 * 4u + 3u], kv.w * dk);
            }
            if (slice == 0u)
                out[(t0 + tt) * p.ldOut + h * p.dv + col] = o;
        }
    }

    [ForceUnroll]
    for (uint i = 0; i < SML_GDN_ROWS; ++i)
        state[(row0 + i) * p.dv + col] = S[i];
}

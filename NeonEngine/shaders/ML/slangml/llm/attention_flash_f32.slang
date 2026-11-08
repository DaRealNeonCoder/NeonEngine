// Flash attention over an fp16 KV cache, for decode and prefill (docs/llm_attention.md).
//
// A workgroup owns one KV head, a group of `tokensPerGroup` query tokens and
// one chunk of KV positions. Its query vectors are every (token, q-head) pair
// that maps to that KV head (GQA), at most SML_ATTN_QG of them, so each K/V row
// is read once for the whole group. Online softmax per vector:
//
//   s = (q . k) * scale            q-head h uses kv head h / (Hq / Hkv)
//   s = softcap * tanh(s / softcap)                  (softcap > 0)
//   masked when pos >= limit(t) or pos < limit(t) - window (window > 0)
//   limit(t) = kvLenBase + t + 1 with SML_ATTN_CAUSAL, else kvLenBase
//   out = sum_p softmax(s)_p v_p, with sinks: denominator += exp(sink[h] - max)
//   out *= sigmoid(gate)                             (SML_ATTN_GATE)
//
// numChunks == 1: the kernel writes the final output.
// numChunks > 1 (flash-decoding): each chunk writes unnormalised partials
// (o, max, sum) to the workspace and sml_llm_attention_combine_fwd_f32 merges
// them. Workspace: O fp32 [numChunks][T*Hq][D], ML float2 [numChunks][T*Hq].
//
// Mapping: 128 threads = 4 groups of 32 lanes. Each 32-lane group walks blocks
// of 32 positions; dims are split D/32 per lane for the dot products and the PV
// accumulation. The 8 per-position dot products are finished with one
// reduce-scatter butterfly (9 shuffles), and fully visible blocks skip masking. The four groups are
// merged through shared memory at the end. Cross-lane ops are 32-lane
// clusters, so 64-wide subgroups work too (two clusters per subgroup).
//
// Grid: x = numChunks, y = ceil(T / tokensPerGroup), z = Hkv.
// Tensor table: 0 Q fp32 [T][qTokenStride] (head h at h*D), 1 Kcache fp16,
//   2 Vcache fp16 ([Hkv][maxCtx][D]), 3 Out fp32 [T][outTokenStride],
//   4 sinks fp32 [Hq], 5 gate fp32, 6 workspace O, 7 workspace ML.
import slangml_common;
import slangml_llm;

static const uint SML_ATTN_QG = 8u;
static const uint SML_ATTN_CAUSAL = 1u;
static const uint SML_ATTN_SINKS = 2u;
static const uint SML_ATTN_GATE = 4u;
static const float SML_NEG_INF = -1.0e30f;  // finite "minus infinity": keeps exp/max NaN-free

struct AttnParams
{
    KernelHeader header;
    uint numQHeads;        // 48
    uint numKVHeads;       // 52
    uint T;                // 56  query tokens
    uint tokensPerGroup;   // 60  (tokensPerGroup * Hq/Hkv <= 8)
    uint qTokenStride;     // 64
    uint maxCtx;           // 68
    uint kvLenBase;        // 72
    uint chunkSize;        // 76  positions per workgroup (multiple of 32)
    uint numChunks;        // 80
    float scale;           // 84
    float softcap;         // 88
    uint window;           // 92
    uint flags;            // 96
    uint gateTokenStride;  // 100
    uint gateHeadStride;   // 104
    uint outTokenStride;   // 108
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlAttnPush;

// Tree merge buffer: two groups x QG vectors x D floats (D <= 256), plus stats.
groupshared float smlAttnO[2 * SML_ATTN_QG * 256];
groupshared float smlAttnM[2 * SML_ATTN_QG];
groupshared float smlAttnL[2 * SML_ATTN_QG];

[ForceInline]
uint sml_attn_limit(AttnParams* p, uint t)
{
    return (p.flags & SML_ATTN_CAUSAL) != 0u ? p.kvLenBase + t + 1u : p.kvLenBase;
}

// Loads DL consecutive fp16 values (DL = 2, 4 or 8) as floats.
[ForceInline]
void sml_attn_load_row<let DL : int>(uint64_t address, out float v[DL])
{
    if (DL == 8)
    {
        uint4 w = sml_ld_u4(reinterpret<uint4*>(address));
        float4 lo, hi;
        sml_unpack_half8(w, lo, hi);
        v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w;
        v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
    }
    else if (DL == 4)
    {
        uint2 w = sml_ld_u2(reinterpret<uint2*>(address));
        float2 a = sml_half2_to_float2(w.x);
        float2 b = sml_half2_to_float2(w.y);
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
    else
    {
        float2 a = sml_half2_to_float2(reinterpret<uint*>(address)[0]);
        v[0] = a.x; v[1] = a.y;
    }
}

// Sums eight per-lane partials across a 32-lane cluster in 9 shuffles
// (instead of 8 x 5). Returns the total for vector (lane >> 2) & 7; lanes
// 4v .. 4v+3 all hold vector v's total.
[ForceInline]
float sml_attn_reduce_scatter8(float s[SML_ATTN_QG], uint lane)
{
    uint L = WaveGetLaneIndex();
    bool b4 = (lane & 16u) != 0u;
    float t[4];
    [ForceUnroll]
    for (uint i = 0; i < 4; ++i)
    {
        float mine = b4 ? s[4u + i] : s[i];
        float other = b4 ? s[i] : s[4u + i];
        t[i] = mine + WaveReadLaneAt(other, L ^ 16u);
    }
    bool b3 = (lane & 8u) != 0u;
    float u[2];
    [ForceUnroll]
    for (uint i = 0; i < 2; ++i)
    {
        float mine = b3 ? t[2u + i] : t[i];
        float other = b3 ? t[i] : t[2u + i];
        u[i] = mine + WaveReadLaneAt(other, L ^ 8u);
    }
    bool b2 = (lane & 4u) != 0u;
    float w = (b2 ? u[1] : u[0]) + WaveReadLaneAt(b2 ? u[0] : u[1], L ^ 4u);
    w += WaveReadLaneAt(w, L ^ 2u);
    w += WaveReadLaneAt(w, L ^ 1u);
    return w;
}

[ForceInline]
void sml_attn_merge(inout float m, inout float l, float m2, float l2, out float a1, out float a2)
{
    float mn = max(m, m2);
    a1 = exp2((m - mn) * SML_LOG2E);
    a2 = exp2((m2 - mn) * SML_LOG2E);
    l = l * a1 + l2 * a2;
    m = mn;
}

[ForceInline]
void sml_attn_main<let D : int>(uint3 gid, uint3 tid)
{
    static const int DL = D / 32;
    AttnParams* p = reinterpret<AttnParams*>(smlAttnPush.headerAddress);
    uint lane = tid.x & 31u;
    uint laneBase = WaveGetLaneIndex() & ~31u;
    uint group = tid.x >> 5u;
    uint chunk = p.header.dispatchBase.x + gid.x;
    uint kvHead = gid.z;
    uint G = p.numQHeads / p.numKVHeads;
    uint t0 = gid.y * p.tokensPerGroup;
    uint nTok = min(p.tokensPerGroup, p.T - t0);
    uint nv = nTok * G;

    uint limitMax = sml_attn_limit(p, t0 + nTok - 1u);
    uint c0 = chunk * p.chunkSize;
    if (c0 >= limitMax)
        return;  // uniform per workgroup; the combine pass never reads this chunk
    uint c1 = min(c0 + p.chunkSize, limitMax);

    float* Q = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u));
    uint64_t kBase = sml_base(slangml_tensor_desc(p.header, 1u), 2u) + uint64_t(kvHead) * p.maxCtx * D * 2u;
    uint64_t vBase = sml_base(slangml_tensor_desc(p.header, 2u), 2u) + uint64_t(kvHead) * p.maxCtx * D * 2u;

    float q[SML_ATTN_QG][DL];
    uint limit[SML_ATTN_QG];
    uint lower[SML_ATTN_QG];
    [ForceUnroll]
    for (uint v = 0; v < SML_ATTN_QG; ++v)
    {
        uint vv = min(v, nv - 1u);
        uint t = t0 + vv / G;
        uint qh = kvHead * G + vv % G;
        float* qp = Q + (t * p.qTokenStride + qh * D + lane * DL);
        [ForceUnroll]
        for (int j = 0; j < DL; ++j)
            q[v][j] = qp[j] * p.scale;
        limit[v] = v < nv ? sml_attn_limit(p, t) : 0u;
        lower[v] = (p.window > 0u && limit[v] > p.window) ? limit[v] - p.window : 0u;
    }

    // Scores use a reduce-scatter layout: after sml_attn_reduce_scatter8, lane L
    // holds the full dot product of vector myV = (L >> 2) & 7, and it keeps
    // positions j = 4 * jj + (L & 3) of the current 32-position block in sc[jj].
    uint myV = (lane >> 2u) & 7u;
    uint myJ = lane & 3u;
    uint myLimit = 0u;
    uint myLower = 0u;
    uint minLimit = 0xFFFFFFFFu;
    uint maxLower = 0u;
    [ForceUnroll]
    for (uint v = 0; v < SML_ATTN_QG; ++v)
    {
        if (v == myV)
        {
            myLimit = limit[v];
            myLower = lower[v];
        }
        if (v < nv)
        {
            minLimit = min(minLimit, limit[v]);
            maxLower = max(maxLower, lower[v]);
        }
    }

    float mRun = SML_NEG_INF;  // running max of vector myV (identical in its 4 lanes)
    float lRun = 0.0f;         // this lane's share of vector myV's running sum
    float o[SML_ATTN_QG][DL];
    [ForceUnroll]
    for (uint v = 0; v < SML_ATTN_QG; ++v)
        [ForceUnroll]
        for (int j = 0; j < DL; ++j)
            o[v][j] = 0.0f;

    bool softcap = p.softcap > 0.0f;
    float invCap = softcap ? 1.0f / p.softcap : 0.0f;
    for (uint b0 = c0 + group * 32u; b0 < c1; b0 += 128u)
    {
        uint count = min(32u, c1 - b0);
        // Blocks entirely inside every vector's visible range skip masking
        // (all of them during decode).
        bool full = count == 32u && b0 + 32u <= minLimit && b0 >= maxLower;
        float sc[8];
        [ForceUnroll]
        for (uint jj = 0; jj < 8; ++jj)
            sc[jj] = SML_NEG_INF;

        [ForceUnroll]
        for (uint j = 0; j < 32u; ++j)
        {
            if (j < count)
            {
                uint pos = b0 + j;
                float k[DL];
                sml_attn_load_row<DL>(kBase + (uint64_t(pos) * D + lane * DL) * 2u, k);
                float part[SML_ATTN_QG];
                [ForceUnroll]
                for (uint v = 0; v < SML_ATTN_QG; ++v)
                {
                    float acc = 0.0f;
                    [ForceUnroll]
                    for (int jj = 0; jj < DL; ++jj)
                        acc = fma(q[v][jj], k[jj], acc);
                    part[v] = acc;
                }
                float w = sml_attn_reduce_scatter8(part, lane);
                if (softcap)
                    w = p.softcap * tanh(w * invCap);
                if (!full)
                    w = (pos < myLimit && pos >= myLower) ? w : SML_NEG_INF;
                if (myJ == (j & 3u))
                    sc[j >> 2u] = w;
            }
        }

        float bm = sc[0];
        [ForceUnroll]
        for (uint jj = 1; jj < 8; ++jj)
            bm = max(bm, sc[jj]);
        bm = max(bm, WaveReadLaneAt(bm, WaveGetLaneIndex() ^ 1u));
        bm = max(bm, WaveReadLaneAt(bm, WaveGetLaneIndex() ^ 2u));
        float mn = max(mRun, bm);
        float corr = exp2((mRun - mn) * SML_LOG2E);
        float pr[8];
        float psum = 0.0f;
        [ForceUnroll]
        for (uint jj = 0; jj < 8; ++jj)
        {
            pr[jj] = sc[jj] <= SML_NEG_INF ? 0.0f : exp2((sc[jj] - mn) * SML_LOG2E);
            psum += pr[jj];
        }
        lRun = lRun * corr + psum;
        mRun = mn;
        [ForceUnroll]
        for (uint v = 0; v < SML_ATTN_QG; ++v)
        {
            float cv = WaveReadLaneAt(corr, laneBase | (v * 4u));
            [ForceUnroll]
            for (int jj = 0; jj < DL; ++jj)
                o[v][jj] *= cv;
        }

        [ForceUnroll]
        for (uint j = 0; j < 32u; ++j)
        {
            if (j < count)
            {
                uint pos = b0 + j;
                float vr[DL];
                sml_attn_load_row<DL>(vBase + (uint64_t(pos) * D + lane * DL) * 2u, vr);
                [ForceUnroll]
                for (uint v = 0; v < SML_ATTN_QG; ++v)
                {
                    float pj = WaveReadLaneAt(pr[j >> 2u], laneBase | (v * 4u + (j & 3u)));
                    [ForceUnroll]
                    for (int jj = 0; jj < DL; ++jj)
                        o[v][jj] = fma(pj, vr[jj], o[v][jj]);
                }
            }
        }
    }
    float lSum = lRun + WaveReadLaneAt(lRun, WaveGetLaneIndex() ^ 1u);
    lSum += WaveReadLaneAt(lSum, WaveGetLaneIndex() ^ 2u);
    float m[SML_ATTN_QG];
    float l[SML_ATTN_QG];
    [ForceUnroll]
    for (uint v = 0; v < SML_ATTN_QG; ++v)
    {
        m[v] = WaveReadLaneAt(mRun, laneBase | (v * 4u));
        l[v] = WaveReadLaneAt(lSum, laneBase | (v * 4u));
    }

    // Tree merge of the four groups: (2,3) -> (0,1), then 1 -> 0.
    [ForceUnroll]
    for (uint stage = 0; stage < 2; ++stage)
    {
        uint writers = stage == 0u ? 2u : 1u;  // groups [writers, 2*writers) write
        if (group >= writers && group < 2u * writers)
        {
            uint slot = group - writers;
            [ForceUnroll]
            for (uint v = 0; v < SML_ATTN_QG; ++v)
            {
                if (lane == 0u)
                {
                    smlAttnM[slot * SML_ATTN_QG + v] = m[v];
                    smlAttnL[slot * SML_ATTN_QG + v] = l[v];
                }
                [ForceUnroll]
                for (int jj = 0; jj < DL; ++jj)
                    smlAttnO[(slot * SML_ATTN_QG + v) * D + lane * DL + jj] = o[v][jj];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (group < writers)
        {
            [ForceUnroll]
            for (uint v = 0; v < SML_ATTN_QG; ++v)
            {
                float a1, a2;
                sml_attn_merge(m[v], l[v], smlAttnM[group * SML_ATTN_QG + v], smlAttnL[group * SML_ATTN_QG + v], a1, a2);
                [ForceUnroll]
                for (int jj = 0; jj < DL; ++jj)
                    o[v][jj] = o[v][jj] * a1 + smlAttnO[(group * SML_ATTN_QG + v) * D + lane * DL + jj] * a2;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (group != 0u)
        return;

    bool final = p.numChunks == 1u;
    float* sinks = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u));
    float* gate = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float* out = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u));
    float* wsO = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u));
    float2* wsML = reinterpret<float2*>(sml_base(slangml_tensor_desc(p.header, 7u), 8u));
    [ForceUnroll]
    for (uint v = 0; v < SML_ATTN_QG; ++v)
    {
        if (v < nv)
        {
            uint t = t0 + v / G;
            uint qh = kvHead * G + v % G;
            if (final)
            {
                float denom = l[v];
                if ((p.flags & SML_ATTN_SINKS) != 0u)
                    denom += exp2((sinks[qh] - m[v]) * SML_LOG2E);
                float invL = denom > 0.0f ? 1.0f / denom : 0.0f;
                [ForceUnroll]
                for (int jj = 0; jj < DL; ++jj)
                {
                    uint d = lane * DL + jj;
                    float r = o[v][jj] * invL;
                    if ((p.flags & SML_ATTN_GATE) != 0u)
                        r *= sml_sigmoid(gate[t * p.gateTokenStride + qh * p.gateHeadStride + d]);
                    out[t * p.outTokenStride + qh * D + d] = r;
                }
            }
            else
            {
                uint vg = t * p.numQHeads + qh;
                uint rowIndex = chunk * p.T * p.numQHeads + vg;
                [ForceUnroll]
                for (int jj = 0; jj < DL; ++jj)
                    wsO[rowIndex * D + lane * DL + jj] = o[v][jj];
                if (lane == 0u)
                    wsML[rowIndex] = float2(m[v], l[v]);
            }
        }
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void sml_llm_attention_fwd_f16kv_d64(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_attn_main<64>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_llm_attention_fwd_f16kv_d128(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_attn_main<128>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_llm_attention_fwd_f16kv_d256(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_attn_main<256>(gid, tid); }

// Split-KV combine: one 64-thread workgroup per (token, q-head) vector.
// Dispatch: x = T * Hq. Same parameter block and tensor table as the main pass,
// followed by headDim (D <= 256).
struct AttnCombineParams
{
    AttnParams attn;
    uint headDim;  // 112
};

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_llm_attention_combine_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    AttnCombineParams* cp = reinterpret<AttnCombineParams*>(smlAttnPush.headerAddress);
    AttnParams* p = &cp.attn;
    uint vg = p.header.dispatchBase.x + gid.x;
    if (vg >= p.T * p.numQHeads)
        return;
    uint D = cp.headDim;
    uint t = vg / p.numQHeads;
    uint qh = vg - t * p.numQHeads;
    uint chunks = (sml_attn_limit(p, t) + p.chunkSize - 1u) / p.chunkSize;
    chunks = min(chunks, p.numChunks);
    float* wsO = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u));
    float2* wsML = reinterpret<float2*>(sml_base(slangml_tensor_desc(p.header, 7u), 8u));
    uint stride = p.T * p.numQHeads;

    float M = SML_NEG_INF;
    for (uint c = 0; c < chunks; ++c)
        M = max(M, wsML[c * stride + vg].x);
    float L = 0.0f;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint c = 0; c < chunks; ++c)
    {
        float2 ml = wsML[c * stride + vg];
        float a = exp2((ml.x - M) * SML_LOG2E);
        L += ml.y * a;
        [ForceUnroll]
        for (uint j = 0; j < 4; ++j)
        {
            uint d = tid.x + 64u * j;
            if (d < D)
                acc[j] += wsO[(c * stride + vg) * D + d] * a;
        }
    }
    if ((p.flags & SML_ATTN_SINKS) != 0u)
        L += exp2((reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u))[qh] - M) * SML_LOG2E);
    float invL = L > 0.0f ? 1.0f / L : 0.0f;
    float* gate = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float* out = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u));
    [ForceUnroll]
    for (uint j = 0; j < 4; ++j)
    {
        uint d = tid.x + 64u * j;
        if (d < D)
        {
            float r = acc[j] * invL;
            if ((p.flags & SML_ATTN_GATE) != 0u)
                r *= sml_sigmoid(gate[t * p.gateTokenStride + qh * p.gateHeadStride + d]);
            out[t * p.outTokenStride + qh * D + d] = r;
        }
    }
}

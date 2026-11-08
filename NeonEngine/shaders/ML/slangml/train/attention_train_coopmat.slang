// Training attention on cooperative matrices (docs/gpt_oss_training.md): the
// same math, parameter block, tensor tables and grids as attention_train_f32.slang
// (causal GQA, gpt-oss sinks in the denominator only, optional sliding window,
// head_dim 64, packed qkv rows, T a multiple of 64), with every product on the
// tensor cores (fp16 inputs, fp32 accumulate, 16x16x16):
//   forward  S = Q K^T, O += P V                      (flash attention, online softmax)
//   dq       S = Q K^T, dP = dO V^T, dQ += dS K
//   dk / dv  S^T = K Q^T, dP^T = V dO^T, dV += P^T dO, dK += dS^T Q
// with P = exp(scale S - L) and dS = P (dP - D). Softmax statistics, L, D, the
// mask and all accumulators stay fp32; q, k, v, dO, P and dS are rounded to
// fp16. dO carries the loss scale, so it must stay below 65504 (an overflow
// becomes inf, the step is skipped and the trainer lowers the scale).
//
// Workgroup: 128 threads = 4 subgroups of 32; each subgroup owns 16 of the 64
// rows (queries for the forward and dq, keys for dk/dv) and keeps its 64-wide
// operand rows as fp16 A fragments in registers. The other side is staged
// 64 rows at a time as fp16 in shared memory ([64][72 halves], the next tile
// prefetched into registers). Accumulator layouts are opaque, so S / dP go
// through a per-subgroup fp32 scratch where lanes apply the mask and the
// softmax (2 lanes per row), and P / dS come back as fp16 A fragments. The
// forward rescales O with a per-row factor loaded as a stride-0 broadcast.
// 16-row blocks with no valid (query, key) pair are skipped.
import slangml_common;
import slangml_llm;
import slangml_train;

using namespace linalg;

typealias FaMatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FaMatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FaMatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

static const uint HD = 64u;
static const uint LDT = 9u;               // fp16 tile row stride in uint4 (72 halves)
static const uint TILE_U4 = 64u * LDT;    // one [64][72] fp16 tile
static const uint SCR_U4 = 272u;          // per-subgroup scratch (uint4)
// Scratch layout (uint4 offsets inside a subgroup's block):
//   forward: S fp32 [16][68] at 0 (row stride 17), then P fp16 [16][72] over it (stride 9)
//   dq/dkv:  S fp32 [16][20] at 0, dP at 80 (stride 5), P fp16 [16][24] at 160, dS at 208 (stride 3)
static const uint SCR_LDS = 17u;
static const uint SCR_LD5 = 5u;
static const uint SCR_DP = 80u;
static const uint SCR_P = 160u;
static const uint SCR_DS = 208u;
static const uint SCR_LD3 = 3u;
static const float FA_NEG = -1.0e30f;

struct TrainAttnParams
{
    KernelHeader header;
    uint T;          // 48
    uint Hq;         // 52
    uint Hkv;        // 56
    uint window;     // 60  0 = full causal
    uint ldQKV;      // 64
    uint ldO;        // 68
    float scale;     // 72
    uint reserved0;  // 76
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTcAttnPush;

groupshared uint4 faT[2u * TILE_U4];   // X tile at 0, Y tile at TILE_U4
groupshared uint4 faScr[4u * SCR_U4];
groupshared float faRowA[64];           // per-row L (dq) / per-column L (dkv) / rescale factor (fwd)
groupshared float faRowB[64];           // D

[ForceInline]
uint fa_half2(float a, float b)
{
    return (f32tof16(a) & 0xFFFFu) | (f32tof16(b) << 16u);
}

[ForceInline]
uint4 fa_half8(float4 lo, float4 hi)
{
    return uint4(fa_half2(lo.x, lo.y), fa_half2(lo.z, lo.w), fa_half2(hi.x, hi.y), fa_half2(hi.z, hi.w));
}

[ForceInline]
bool fa_valid(uint i, uint j, uint window)
{
    return j <= i && (window == 0u || i - j < window);
}

// Whether any query in [qa, qa + 16) may attend to any key in [ka, ka + 16).
[ForceInline]
bool fa_any(uint qa, uint ka, uint window)
{
    if (qa + 15u < ka)
        return false;
    return window == 0u || qa < ka + 15u + window;
}

// One 64 x 64 fp32 tile in registers: 4 octets (8 floats) per thread;
// octet l = tid + 128 s covers row l / 8, columns 8 (l % 8).
struct FaRegs
{
    float4 v[8];
};

[ForceInline]
void fa_fetch(float* base, uint ld, uint row0, uint col, uint tid, out FaRegs r)
{
    [ForceUnroll]
    for (uint s = 0; s < 4u; ++s)
    {
        uint l = tid + 128u * s;
        float4* src = reinterpret<float4*>(base + ((row0 + (l >> 3u)) * ld + col + (l & 7u) * 8u));
        r.v[2u * s] = sml_ld_f4(src);
        r.v[2u * s + 1u] = sml_ld_f4(src + 1);
    }
}

[ForceInline]
void fa_put(uint off, uint tid, FaRegs r)
{
    [ForceUnroll]
    for (uint s = 0; s < 4u; ++s)
    {
        uint l = tid + 128u * s;
        faT[off + (l >> 3u) * LDT + (l & 7u)] = fa_half8(r.v[2u * s], r.v[2u * s + 1u]);
    }
}

[ForceInline]
void fa_scale_rows(inout FaMatC m, FaMatC f)
{
    for (int e = 0; e < int(FaMatC.GetLength()); ++e)  // a driver constant, not known to Slang
        m[e] = m[e] * f[e];
}

// ---------------------------------------------------------------------------
// Forward. Grid: x = T / 64, y = Hq, z = batch.
// Tensor table: 0 qkv, 1 sinks [Hq], 2 O [rows][ldO], 3 L [rows][Hq].
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void sml_train_attn_fwd_tc(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTcAttnPush.headerAddress);
    uint tid = tid3.x;
    uint warp = tid >> 5u;
    uint lane = tid & 31u;
    uint h = gid.y;
    uint b = gid.z;
    uint kvh = h / (p.Hq / p.Hkv);
    uint i0 = gid.x * 64u;
    uint qa = i0 + warp * 16u;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;
    uint scr = warp * SCR_U4;

    FaRegs ra, rb;
    fa_fetch(qkv, p.ldQKV, i0, h * HD, tid, ra);
    fa_put(0u, tid, ra);
    GroupMemoryBarrierWithGroupSync();
    FaMatA qf[4];
    [ForceUnroll]
    for (uint k = 0; k < 4u; ++k)
        qf[k] = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faT, warp * 16u * LDT + 2u * k, LDT);
    GroupMemoryBarrierWithGroupSync();

    uint kStart = (p.window != 0u && i0 + 1u > p.window) ? ((i0 + 1u - p.window) & ~63u) : 0u;
    uint kEnd = i0 + 64u;
    fa_fetch(qkv, p.ldQKV, kStart, colK, tid, ra);
    fa_fetch(qkv, p.ldQKV, kStart, colV, tid, rb);

    // Softmax state: lanes 2r and 2r + 1 own row r (32 keys of the tile each).
    uint r = lane >> 1u;
    uint hc = lane & 1u;
    uint i = qa + r;
    float m = sml_tf(p.header, 1u)[h];  // the sink: exp(sink - m) = 1 starts l
    float l = 1.0f;
    FaMatC o[4];
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
        o[nd] = FaMatC(0.0f);

    for (uint kt = kStart; kt < kEnd; kt += 64u)
    {
        fa_put(0u, tid, ra);
        fa_put(TILE_U4, tid, rb);
        GroupMemoryBarrierWithGroupSync();
        if (kt + 64u < kEnd)
        {
            fa_fetch(qkv, p.ldQKV, kt + 64u, colK, tid, ra);
            fa_fetch(qkv, p.ldQKV, kt + 64u, colV, tid, rb);
        }

        bool act[4];
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n)
        {
            act[n] = fa_any(qa, kt + 16u * n, p.window);
            if (act[n])
            {
                FaMatC s = FaMatC(0.0f);
                [ForceUnroll]
                for (uint k = 0; k < 4u; ++k)
                    s = coopMatMulAdd<float, false>(
                        qf[k], FaMatB.Load<CoopMatMatrixLayout.ColumnMajor>(faT, 16u * n * LDT + 2u * k, LDT), s);
                s.Store<CoopMatMatrixLayout.RowMajor>(faScr, scr + 4u * n, SCR_LDS);
            }
        }
        GroupMemoryBarrierWithWaveSync();

        float sv[32];
        [ForceUnroll]
        for (uint q4 = 0; q4 < 8u; ++q4)
        {
            uint4 u = faScr[scr + r * SCR_LDS + hc * 8u + q4];
            sv[4u * q4] = asfloat(u.x);
            sv[4u * q4 + 1u] = asfloat(u.y);
            sv[4u * q4 + 2u] = asfloat(u.z);
            sv[4u * q4 + 3u] = asfloat(u.w);
        }
        float mx = m;
        uint jb = kt + hc * 32u;
        [ForceUnroll]
        for (uint c = 0; c < 32u; ++c)
        {
            sv[c] = fa_valid(i, jb + c, p.window) ? sv[c] * p.scale : FA_NEG;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, WaveReadLaneAt(mx, lane ^ 1u));
        float alpha = exp(m - mx);
        float sum = 0.0f;
        [ForceUnroll]
        for (uint c = 0; c < 32u; ++c)
        {
            sv[c] = exp(sv[c] - mx);
            sum += sv[c];
        }
        sum += WaveReadLaneAt(sum, lane ^ 1u);
        l = l * alpha + sum;
        m = mx;
        GroupMemoryBarrierWithWaveSync();  // S reads done before P overwrites it
        [ForceUnroll]
        for (uint q8 = 0; q8 < 4u; ++q8)
            faScr[scr + r * LDT + hc * 4u + q8] =
                fa_half8(float4(sv[8u * q8], sv[8u * q8 + 1u], sv[8u * q8 + 2u], sv[8u * q8 + 3u]),
                         float4(sv[8u * q8 + 4u], sv[8u * q8 + 5u], sv[8u * q8 + 6u], sv[8u * q8 + 7u]));
        if (hc == 0u)
            faRowA[warp * 16u + r] = alpha;
        GroupMemoryBarrierWithWaveSync();

        FaMatC af = FaMatC.Load<CoopMatMatrixLayout.ColumnMajor>(faRowA, warp * 16u, 0u);
        [ForceUnroll]
        for (uint nd = 0; nd < 4u; ++nd)
            fa_scale_rows(o[nd], af);
        [ForceUnroll]
        for (uint k = 0; k < 4u; ++k)
        {
            if (act[k])
            {
                FaMatA pf = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faScr, scr + 2u * k, LDT);
                [ForceUnroll]
                for (uint nd = 0; nd < 4u; ++nd)
                    o[nd] = coopMatMulAdd<float, false>(
                        pf, FaMatB.Load<CoopMatMatrixLayout.RowMajor>(faT, TILE_U4 + 16u * k * LDT + 2u * nd, LDT), o[nd]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (hc == 0u)
        faRowA[warp * 16u + r] = 1.0f / l;
    GroupMemoryBarrierWithWaveSync();
    FaMatC inv = FaMatC.Load<CoopMatMatrixLayout.ColumnMajor>(faRowA, warp * 16u, 0u);
    float* O = sml_tf(p.header, 2u) + (b * p.T + qa) * p.ldO + h * HD;
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
    {
        fa_scale_rows(o[nd], inv);
        o[nd].Store<CoopMatMatrixLayout.RowMajor>(O + 16u * nd, 0u, p.ldO);
    }
    if (hc == 0u)
        sml_tf(p.header, 3u)[(b * p.T + i) * p.Hq + h] = m + log(l);
}

// ---------------------------------------------------------------------------
// Backward, query side: dq, D and the per-row sink-gradient terms.
// Grid: x = T / 64, y = Hq, z = batch.
// Tensor table: 0 qkv, 1 sinks, 2 O, 3 dO [rows][ldO], 4 L, 5 dqkv (q part
// written), 6 D [rows][Hq], 7 sinkTerm [rows][Hq] (= -exp(sink - L) D).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void sml_train_attn_bwd_dq_tc(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTcAttnPush.headerAddress);
    uint tid = tid3.x;
    uint warp = tid >> 5u;
    uint lane = tid & 31u;
    uint h = gid.y;
    uint b = gid.z;
    uint kvh = h / (p.Hq / p.Hkv);
    uint i0 = gid.x * 64u;
    uint qa = i0 + warp * 16u;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;
    uint scr = warp * SCR_U4;

    // Stage q and dO (fp16), and compute D = dO . O in fp32 (8 lanes per row).
    {
        float* Og = sml_tf(p.header, 2u);
        float* dOg = sml_tf(p.header, 3u);
        float sink = sml_tf(p.header, 1u)[h];
        [ForceUnroll]
        for (uint s = 0; s < 4u; ++s)
        {
            uint l = tid + 128u * s;
            uint row = l >> 3u;
            uint oc = (l & 7u) * 8u;
            uint n = b * p.T + i0 + row;
            float4* qs = reinterpret<float4*>(qkv + ((i0 + row) * p.ldQKV + h * HD + oc));
            float4* ds = reinterpret<float4*>(dOg + (n * p.ldO + h * HD + oc));
            float4* os = reinterpret<float4*>(Og + (n * p.ldO + h * HD + oc));
            float4 d0 = sml_ld_f4(ds), d1 = sml_ld_f4(ds + 1);
            float4 o0 = sml_ld_f4(os), o1 = sml_ld_f4(os + 1);
            faT[row * LDT + (l & 7u)] = fa_half8(sml_ld_f4(qs), sml_ld_f4(qs + 1));
            faT[TILE_U4 + row * LDT + (l & 7u)] = fa_half8(d0, d1);
            float Dv = sml_cluster_sum(dot(d0, o0) + dot(d1, o1), 8u);
            if ((l & 7u) == 0u)
            {
                float Lv = sml_tf(p.header, 4u)[n * p.Hq + h];
                faRowA[row] = Lv;
                faRowB[row] = Dv;
                sml_tf(p.header, 6u)[n * p.Hq + h] = Dv;
                sml_tf(p.header, 7u)[n * p.Hq + h] = -exp(sink - Lv) * Dv;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    FaMatA qf[4], df[4];
    [ForceUnroll]
    for (uint k = 0; k < 4u; ++k)
    {
        qf[k] = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faT, warp * 16u * LDT + 2u * k, LDT);
        df[k] = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faT, TILE_U4 + warp * 16u * LDT + 2u * k, LDT);
    }
    GroupMemoryBarrierWithGroupSync();

    uint kStart = (p.window != 0u && i0 + 1u > p.window) ? ((i0 + 1u - p.window) & ~63u) : 0u;
    uint kEnd = i0 + 64u;
    FaRegs ra, rb;
    fa_fetch(qkv, p.ldQKV, kStart, colK, tid, ra);
    fa_fetch(qkv, p.ldQKV, kStart, colV, tid, rb);

    uint r = lane >> 1u;
    uint hc = lane & 1u;
    uint i = qa + r;
    float Lr = faRowA[warp * 16u + r];
    float Dr = faRowB[warp * 16u + r];
    FaMatC dq[4];
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
        dq[nd] = FaMatC(0.0f);

    for (uint kt = kStart; kt < kEnd; kt += 64u)
    {
        fa_put(0u, tid, ra);
        fa_put(TILE_U4, tid, rb);
        GroupMemoryBarrierWithGroupSync();
        if (kt + 64u < kEnd)
        {
            fa_fetch(qkv, p.ldQKV, kt + 64u, colK, tid, ra);
            fa_fetch(qkv, p.ldQKV, kt + 64u, colV, tid, rb);
        }
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n)
        {
            uint ka = kt + 16u * n;
            if (!fa_any(qa, ka, p.window))
                continue;
            FaMatC s = FaMatC(0.0f);
            FaMatC dp = FaMatC(0.0f);
            [ForceUnroll]
            for (uint k = 0; k < 4u; ++k)
            {
                s = coopMatMulAdd<float, false>(
                    qf[k], FaMatB.Load<CoopMatMatrixLayout.ColumnMajor>(faT, 16u * n * LDT + 2u * k, LDT), s);
                dp = coopMatMulAdd<float, false>(
                    df[k], FaMatB.Load<CoopMatMatrixLayout.ColumnMajor>(faT, TILE_U4 + 16u * n * LDT + 2u * k, LDT), dp);
            }
            s.Store<CoopMatMatrixLayout.RowMajor>(faScr, scr, SCR_LD5);
            dp.Store<CoopMatMatrixLayout.RowMajor>(faScr, scr + SCR_DP, SCR_LD5);
            GroupMemoryBarrierWithWaveSync();

            float dsv[8];
            [ForceUnroll]
            for (uint q4 = 0; q4 < 2u; ++q4)
            {
                uint4 su = faScr[scr + r * SCR_LD5 + hc * 2u + q4];
                uint4 pu = faScr[scr + SCR_DP + r * SCR_LD5 + hc * 2u + q4];
                float sa[4] = { asfloat(su.x), asfloat(su.y), asfloat(su.z), asfloat(su.w) };
                float pa[4] = { asfloat(pu.x), asfloat(pu.y), asfloat(pu.z), asfloat(pu.w) };
                [ForceUnroll]
                for (uint e = 0; e < 4u; ++e)
                {
                    uint j = ka + hc * 8u + 4u * q4 + e;
                    float pr = fa_valid(i, j, p.window) ? exp(sa[e] * p.scale - Lr) : 0.0f;
                    dsv[4u * q4 + e] = pr * (pa[e] - Dr);
                }
            }
            faScr[scr + SCR_DS + r * SCR_LD3 + hc] =
                fa_half8(float4(dsv[0], dsv[1], dsv[2], dsv[3]), float4(dsv[4], dsv[5], dsv[6], dsv[7]));
            GroupMemoryBarrierWithWaveSync();

            FaMatA dsf = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faScr, scr + SCR_DS, SCR_LD3);
            [ForceUnroll]
            for (uint nd = 0; nd < 4u; ++nd)
                dq[nd] = coopMatMulAdd<float, false>(
                    dsf, FaMatB.Load<CoopMatMatrixLayout.RowMajor>(faT, 16u * n * LDT + 2u * nd, LDT), dq[nd]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float* dqg = sml_tf(p.header, 5u) + (b * p.T + qa) * p.ldQKV + h * HD;
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
    {
        FaMatC t = dq[nd] * p.scale;
        t.Store<CoopMatMatrixLayout.RowMajor>(dqg + 16u * nd, 0u, p.ldQKV);
    }
}

// ---------------------------------------------------------------------------
// Backward, key side: dk and dv for one kv head, summed over its query heads.
// Grid: x = T / 64 (key blocks), y = Hkv, z = batch.
// Tensor table: 0 qkv, 1 dO, 2 L, 3 D, 4 dqkv (k and v parts written).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void sml_train_attn_bwd_dkv_tc(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTcAttnPush.headerAddress);
    uint tid = tid3.x;
    uint warp = tid >> 5u;
    uint lane = tid & 31u;
    uint kvh = gid.y;
    uint b = gid.z;
    uint G = p.Hq / p.Hkv;
    uint j0 = gid.x * 64u;
    uint ka = j0 + warp * 16u;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    float* dOb = sml_tf(p.header, 1u) + b * p.T * p.ldO;
    float* Lb = sml_tf(p.header, 2u) + b * p.T * p.Hq;
    float* Db = sml_tf(p.header, 3u) + b * p.T * p.Hq;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;
    uint scr = warp * SCR_U4;

    FaRegs ra, rb;
    fa_fetch(qkv, p.ldQKV, j0, colK, tid, ra);
    fa_fetch(qkv, p.ldQKV, j0, colV, tid, rb);
    fa_put(0u, tid, ra);
    fa_put(TILE_U4, tid, rb);
    GroupMemoryBarrierWithGroupSync();
    FaMatA kf[4], vf[4];
    [ForceUnroll]
    for (uint k = 0; k < 4u; ++k)
    {
        kf[k] = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faT, warp * 16u * LDT + 2u * k, LDT);
        vf[k] = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faT, TILE_U4 + warp * 16u * LDT + 2u * k, LDT);
    }
    GroupMemoryBarrierWithGroupSync();

    uint qEnd = p.T;
    if (p.window != 0u)
        qEnd = min(p.T, (j0 + 64u + p.window - 1u + 63u) & ~63u);
    uint nq = (qEnd - j0) >> 6u;
    uint items = G * nq;

    // Item it = (query head g = it / nq, query tile qt = j0 + 64 (it % nq)).
    float lpre = 0.0f, dpre = 0.0f;
    {
        uint h = kvh * G;
        fa_fetch(qkv, p.ldQKV, j0, h * HD, tid, ra);
        fa_fetch(dOb, p.ldO, j0, h * HD, tid, rb);
        if (tid < 64u)
        {
            lpre = Lb[(j0 + tid) * p.Hq + h];
            dpre = Db[(j0 + tid) * p.Hq + h];
        }
    }

    uint r = lane >> 1u;
    uint hc = lane & 1u;
    uint j = ka + r;
    FaMatC dk[4], dv[4];
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
    {
        dk[nd] = FaMatC(0.0f);
        dv[nd] = FaMatC(0.0f);
    }

    for (uint it = 0; it < items; ++it)
    {
        uint qt = j0 + 64u * (it % nq);
        fa_put(0u, tid, ra);
        fa_put(TILE_U4, tid, rb);
        if (tid < 64u)
        {
            faRowA[tid] = lpre;
            faRowB[tid] = dpre;
        }
        GroupMemoryBarrierWithGroupSync();
        if (it + 1u < items)
        {
            uint h = kvh * G + (it + 1u) / nq;
            uint qn = j0 + 64u * ((it + 1u) % nq);
            fa_fetch(qkv, p.ldQKV, qn, h * HD, tid, ra);
            fa_fetch(dOb, p.ldO, qn, h * HD, tid, rb);
            if (tid < 64u)
            {
                lpre = Lb[(qn + tid) * p.Hq + h];
                dpre = Db[(qn + tid) * p.Hq + h];
            }
        }
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n)
        {
            uint qa = qt + 16u * n;
            if (!fa_any(qa, ka, p.window))
                continue;
            FaMatC s = FaMatC(0.0f);
            FaMatC dp = FaMatC(0.0f);
            [ForceUnroll]
            for (uint k = 0; k < 4u; ++k)
            {
                s = coopMatMulAdd<float, false>(
                    kf[k], FaMatB.Load<CoopMatMatrixLayout.ColumnMajor>(faT, 16u * n * LDT + 2u * k, LDT), s);
                dp = coopMatMulAdd<float, false>(
                    vf[k], FaMatB.Load<CoopMatMatrixLayout.ColumnMajor>(faT, TILE_U4 + 16u * n * LDT + 2u * k, LDT), dp);
            }
            s.Store<CoopMatMatrixLayout.RowMajor>(faScr, scr, SCR_LD5);
            dp.Store<CoopMatMatrixLayout.RowMajor>(faScr, scr + SCR_DP, SCR_LD5);
            GroupMemoryBarrierWithWaveSync();

            // Row r = key j, columns = queries qa + 8 hc + e.
            float pv[8], dsv[8];
            [ForceUnroll]
            for (uint q4 = 0; q4 < 2u; ++q4)
            {
                uint4 su = faScr[scr + r * SCR_LD5 + hc * 2u + q4];
                uint4 pu = faScr[scr + SCR_DP + r * SCR_LD5 + hc * 2u + q4];
                float sa[4] = { asfloat(su.x), asfloat(su.y), asfloat(su.z), asfloat(su.w) };
                float pa[4] = { asfloat(pu.x), asfloat(pu.y), asfloat(pu.z), asfloat(pu.w) };
                [ForceUnroll]
                for (uint e = 0; e < 4u; ++e)
                {
                    uint c = 16u * n + hc * 8u + 4u * q4 + e;
                    float pr = fa_valid(qt + c, j, p.window) ? exp(sa[e] * p.scale - faRowA[c]) : 0.0f;
                    pv[4u * q4 + e] = pr;
                    dsv[4u * q4 + e] = pr * (pa[e] - faRowB[c]);
                }
            }
            faScr[scr + SCR_P + r * SCR_LD3 + hc] =
                fa_half8(float4(pv[0], pv[1], pv[2], pv[3]), float4(pv[4], pv[5], pv[6], pv[7]));
            faScr[scr + SCR_DS + r * SCR_LD3 + hc] =
                fa_half8(float4(dsv[0], dsv[1], dsv[2], dsv[3]), float4(dsv[4], dsv[5], dsv[6], dsv[7]));
            GroupMemoryBarrierWithWaveSync();

            FaMatA pf = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faScr, scr + SCR_P, SCR_LD3);
            FaMatA dsf = FaMatA.Load<CoopMatMatrixLayout.RowMajor>(faScr, scr + SCR_DS, SCR_LD3);
            [ForceUnroll]
            for (uint nd = 0; nd < 4u; ++nd)
            {
                dv[nd] = coopMatMulAdd<float, false>(
                    pf, FaMatB.Load<CoopMatMatrixLayout.RowMajor>(faT, TILE_U4 + 16u * n * LDT + 2u * nd, LDT), dv[nd]);
                dk[nd] = coopMatMulAdd<float, false>(
                    dsf, FaMatB.Load<CoopMatMatrixLayout.RowMajor>(faT, 16u * n * LDT + 2u * nd, LDT), dk[nd]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float* dqkv = sml_tf(p.header, 4u) + (b * p.T + ka) * p.ldQKV;
    [ForceUnroll]
    for (uint nd = 0; nd < 4u; ++nd)
    {
        FaMatC t = dk[nd] * p.scale;
        t.Store<CoopMatMatrixLayout.RowMajor>(dqkv + colK + 16u * nd, 0u, p.ldQKV);
        dv[nd].Store<CoopMatMatrixLayout.RowMajor>(dqkv + colV + 16u * nd, 0u, p.ldQKV);
    }
}

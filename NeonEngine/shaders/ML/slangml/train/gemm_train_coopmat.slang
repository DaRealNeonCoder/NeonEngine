// Training GEMM on cooperative matrices (docs/gpt_oss_training.md): the same
// products, parameter block and modes as gemm_train_f32.slang, but A and B are
// rounded to fp16 in shared memory and multiplied on the tensor cores
// (16x16x16, fp32 accumulate). Inputs must stay below 65504 in magnitude; the
// trainer scales the loss so gradients do not underflow (dynamic loss scaling).
//
//   C[m][n] = alpha * sum_k A(m, k) B(k, n)  (+ bias[n]) (+ R[m][n]) (+ C[m][n])
//
// One entry point per operand layout, picked by the host from the strides:
//   A: _k  k-contiguous (sAk = 1, row stride sAm)    _m  m-contiguous (sAm = 1, sAk)
//   B: _k  k-contiguous (sBk = 1, row stride sBn)    _n  n-contiguous (sBn = 1, sBk)
//   sml_train_gemm_tc_{A}{B}: kk = forward (X W^T), kn = dX (dY W), mn = dW (dY^T X).
// Requirements (checked by the host, else it uses the fp32 kernel): the
// contiguous strides, ldC, strideBG and strideCG are multiples of 4 and the A,
// B, C base addresses are 16-byte aligned. Any M, N, K (edges are zero-filled).
//
// Modes (p.mode), as in the fp32 kernel:
//   0 plain, grid z = split-K index (k in [z * kChunk, ...), C + z * strideCG)
//   1 M-grouped: g = table[tileM] (SML_INVALID = unused tile), B += g * strideBG,
//     bias += g * strideBiasG
//   2 K-grouped: grid z = group g, k in [table[g], table[g+1]), C += g * strideCG
//
// Tile: 128 x 128 x 32 per 256-thread workgroup (8 subgroups of 32 lanes as
// 4 (M) x 2 (N), 32 x 64 outputs each). Shared tiles are fp16 in two stages:
// a k-contiguous operand as [128][40] halves, an outer-contiguous one as
// [32][136] halves (both strides keep the fragment loads conflict free). The
// next K tile is fetched into registers while the current one is multiplied.
// Full tiles (alpha = 1) preload C / residual / bias into the accumulators and
// store them directly; edge tiles go through a shared-memory epilogue.
// Grid: x = ceil(N / 128), y = ceil(M / 128).
// Tensor table: 0 A, 1 B, 2 C, 3 bias [N], 4 R (ldC), 5 table (uint).
import slangml_common;
import slangml_llm;
import slangml_train;

using namespace linalg;

typealias TcMatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias TcMatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias TcMatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

static const uint TC_FLAG_BIAS = 1u;
static const uint TC_FLAG_RESID = 2u;
static const uint TC_FLAG_ACCUM = 4u;

static const uint TC_BK = 32u;
static const uint TC_LDK = 5u;      // [128][40 halves]: row stride in uint4 (8 halves)
static const uint TC_LDO = 17u;     // [32][136 halves]
static const uint TC_STAGE = 640u;  // uint4 per stage: max(128 * 5, 32 * 17)

struct TrainGemmParams
{
    KernelHeader header;
    uint M;            // 48
    uint N;            // 52
    uint K;            // 56
    uint flags;        // 60
    uint sAm;          // 64
    uint sAk;          // 68
    uint sBk;          // 72
    uint sBn;          // 76
    uint ldC;          // 80
    uint mode;         // 84
    uint kChunk;       // 88
    uint strideBG;     // 92
    uint strideBiasG;  // 96
    uint strideCG;     // 100
    float alpha;       // 104
    uint reserved;     // 108
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTcGemmPush;

groupshared uint4 tcA[2 * TC_STAGE];
groupshared uint4 tcB[2 * TC_STAGE];

[ForceInline]
uint tc_half2(float a, float b)
{
    return (f32tof16(a) & 0xFFFFu) | (f32tof16(b) << 16u);
}

[ForceInline]
uint4 tc_half8(float4 lo, float4 hi)
{
    return uint4(tc_half2(lo.x, lo.y), tc_half2(lo.z, lo.w), tc_half2(hi.x, hi.y), tc_half2(hi.z, hi.w));
}

// 8 consecutive floats at p; element j is read only when j < nvalid.
[ForceInline]
void tc_load8(float* p, uint nvalid, out float4 lo, out float4 hi)
{
    if (nvalid >= 8u)
    {
        lo = sml_ld_f4(reinterpret<float4*>(p));
        hi = sml_ld_f4(reinterpret<float4*>(p) + 1);
    }
    else
    {
        float v[8];
        [ForceUnroll]
        for (uint j = 0; j < 8u; ++j)
            v[j] = j < nvalid ? p[j] : 0.0f;
        lo = float4(v[0], v[1], v[2], v[3]);
        hi = float4(v[4], v[5], v[6], v[7]);
    }
}

// One operand's K tile in registers: two octets (8 floats along the contiguous
// dimension) per thread. Octet l = tid + 256 i:
//   k-contiguous      row l / 4 of 128, k 8 (l % 4)       -> shared [row][k]
//   outer-contiguous  k l / 16 of 32, outer 8 (l % 16)    -> shared [k][outer]
struct TcOperand
{
    float4 v[4];
};

[ForceInline]
void tc_fetch<let KCONTIG : int>(float* base, uint ld, uint o0, uint outerLim, uint k0, uint kEnd, uint tid,
                                 out TcOperand r)
{
    [ForceUnroll]
    for (uint i = 0; i < 2u; ++i)
    {
        uint l = tid + 256u * i;
        uint outer, k;
        if (KCONTIG != 0)
        {
            outer = o0 + (l >> 2u);
            k = k0 + (l & 3u) * 8u;
            uint n = (outer < outerLim && k < kEnd) ? min(8u, kEnd - k) : 0u;
            tc_load8(base + (outer * ld + k), n, r.v[2 * i], r.v[2 * i + 1]);
        }
        else
        {
            k = k0 + (l >> 4u);
            outer = o0 + (l & 15u) * 8u;
            uint n = (k < kEnd && outer < outerLim) ? min(8u, outerLim - outer) : 0u;
            tc_load8(base + (k * ld + outer), n, r.v[2 * i], r.v[2 * i + 1]);
        }
    }
}

[ForceInline]
void tc_store<let KCONTIG : int>(uint stage, uint tid, TcOperand r, bool toA)
{
    [ForceUnroll]
    for (uint i = 0; i < 2u; ++i)
    {
        uint l = tid + 256u * i;
        uint idx = KCONTIG != 0 ? (l >> 2u) * TC_LDK + (l & 3u) : (l >> 4u) * TC_LDO + (l & 15u);
        uint4 h = tc_half8(r.v[2 * i], r.v[2 * i + 1]);
        if (toA)
            tcA[stage * TC_STAGE + idx] = h;
        else
            tcB[stage * TC_STAGE + idx] = h;
    }
}

[ForceInline]
void tc_gemm_main<let AK : int, let BK : int>(uint3 gid, uint3 tid3)
{
    TrainGemmParams* p = reinterpret<TrainGemmParams*>(smlTcGemmPush.headerAddress);
    uint tid = tid3.x;
    uint warp = tid >> 5u;
    uint lane = tid & 31u;
    uint wm = warp >> 1u;  // 0..3: 32 rows each
    uint wn = warp & 1u;   // 0..1: 64 columns each
    uint tileN = gid.x;
    uint tileM = gid.y;
    uint z = gid.z;

    float* A = sml_tf(p.header, 0u);
    float* B = sml_tf(p.header, 1u);
    float* C = sml_tf(p.header, 2u);
    float* bias = sml_tf(p.header, 3u);
    float* R = sml_tf(p.header, 4u);
    uint* table = sml_tu(p.header, 5u);

    uint kBegin = 0u;
    uint kEnd = p.K;
    if (p.mode == 0u)
    {
        kBegin = z * p.kChunk;
        kEnd = min(p.K, kBegin + p.kChunk);
        C = C + (z * p.strideCG);
    }
    else if (p.mode == 1u)
    {
        uint g = table[tileM];
        if (g == SML_INVALID)
            return;  // uniform per workgroup
        B = B + (g * p.strideBG);
        bias = bias + (g * p.strideBiasG);
    }
    else
    {
        kBegin = table[z];
        kEnd = table[z + 1u];
        C = C + (z * p.strideCG);
        if (kBegin >= kEnd && (p.flags & TC_FLAG_ACCUM) != 0u)
            return;
    }
    uint m0 = tileM * 128u;
    uint n0 = tileN * 128u;
    uint ldA = AK != 0 ? p.sAm : p.sAk;
    uint ldB = BK != 0 ? p.sBn : p.sBk;

    // Full tiles with alpha = 1 start the accumulators at C (accumulate) + R
    // (residual) + bias (a stride-0 load broadcasts the bias row) and store
    // them straight to global memory; other tiles use the shared-memory epilogue.
    bool useBias = (p.flags & TC_FLAG_BIAS) != 0u;
    bool useResid = (p.flags & TC_FLAG_RESID) != 0u;
    bool useAccum = (p.flags & TC_FLAG_ACCUM) != 0u;
    bool direct = m0 + 128u <= p.M && n0 + 128u <= p.N && p.alpha == 1.0f;

    TcMatC acc[2][4];
    [ForceUnroll]
    for (uint i = 0; i < 2u; ++i)
        [ForceUnroll]
        for (uint j = 0; j < 4u; ++j)
        {
            acc[i][j] = TcMatC(0.0f);
            if (direct)
            {
                uint rm = m0 + wm * 32u + i * 16u;
                uint cn = n0 + wn * 64u + j * 16u;
                if (useAccum)
                    acc[i][j] = acc[i][j] + TcMatC.Load<CoopMatMatrixLayout.RowMajor>(C + (rm * p.ldC + cn), 0u, p.ldC);
                if (useResid)
                    acc[i][j] = acc[i][j] + TcMatC.Load<CoopMatMatrixLayout.RowMajor>(R + (rm * p.ldC + cn), 0u, p.ldC);
                if (useBias)
                    acc[i][j] = acc[i][j] + TcMatC.Load<CoopMatMatrixLayout.RowMajor>(bias + cn, 0u, 0u);
            }
        }

    uint nk = kEnd > kBegin ? (kEnd - kBegin + TC_BK - 1u) / TC_BK : 0u;
    TcOperand ra, rb;
    if (nk > 0u)
    {
        tc_fetch<AK>(A, ldA, m0, p.M, kBegin, kEnd, tid, ra);
        tc_fetch<BK>(B, ldB, n0, p.N, kBegin, kEnd, tid, rb);
        tc_store<AK>(0u, tid, ra, true);
        tc_store<BK>(0u, tid, rb, false);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint kt = 0; kt < nk; ++kt)
    {
        uint cur = kt & 1u;
        bool more = kt + 1u < nk;
        if (more)
        {
            uint kn = kBegin + (kt + 1u) * TC_BK;
            tc_fetch<AK>(A, ldA, m0, p.M, kn, kEnd, tid, ra);
            tc_fetch<BK>(B, ldB, n0, p.N, kn, kEnd, tid, rb);
        }
        uint sa = cur * TC_STAGE;
        [ForceUnroll]
        for (uint ks = 0; ks < 2u; ++ks)
        {
            TcMatA fa[2];
            TcMatB fb[4];
            [ForceUnroll]
            for (uint i = 0; i < 2u; ++i)
            {
                uint r0 = wm * 32u + i * 16u;
                if (AK != 0)
                    fa[i] = TcMatA.Load<CoopMatMatrixLayout.RowMajor>(tcA, sa + r0 * TC_LDK + ks * 2u, TC_LDK);
                else
                    fa[i] = TcMatA.Load<CoopMatMatrixLayout.ColumnMajor>(tcA, sa + ks * 16u * TC_LDO + (r0 >> 3u), TC_LDO);
            }
            [ForceUnroll]
            for (uint j = 0; j < 4u; ++j)
            {
                uint c0 = wn * 64u + j * 16u;
                if (BK != 0)
                    fb[j] = TcMatB.Load<CoopMatMatrixLayout.ColumnMajor>(tcB, sa + c0 * TC_LDK + ks * 2u, TC_LDK);
                else
                    fb[j] = TcMatB.Load<CoopMatMatrixLayout.RowMajor>(tcB, sa + ks * 16u * TC_LDO + (c0 >> 3u), TC_LDO);
            }
            [ForceUnroll]
            for (uint i = 0; i < 2u; ++i)
                [ForceUnroll]
                for (uint j = 0; j < 4u; ++j)
                    acc[i][j] = coopMatMulAdd<float, false>(fa[i], fb[j], acc[i][j]);
        }
        if (more)
        {
            tc_store<AK>(cur ^ 1u, tid, ra, true);
            tc_store<BK>(cur ^ 1u, tid, rb, false);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Epilogue.
    [ForceUnroll]
    for (uint i = 0; i < 2u; ++i)
    {
        [ForceUnroll]
        for (uint j = 0; j < 4u; ++j)
        {
            uint rm = m0 + wm * 32u + i * 16u;
            uint cn = n0 + wn * 64u + j * 16u;
            if (direct)
            {
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(C + (rm * p.ldC + cn), 0u, p.ldC);
            }
            else
            {
                // Per-subgroup 16 x 16 fp32 scratch in the idle B stages (64 uint4, row stride 4).
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(tcB, warp * 64u, 4u);
                uint r = lane >> 1u;
                uint c8 = (lane & 1u) * 8u;
                uint gm = rm + r;
                uint4 q0 = tcB[warp * 64u + r * 4u + (c8 >> 2u)];
                uint4 q1 = tcB[warp * 64u + r * 4u + (c8 >> 2u) + 1u];
                float vals[8] = { asfloat(q0.x), asfloat(q0.y), asfloat(q0.z), asfloat(q0.w),
                                  asfloat(q1.x), asfloat(q1.y), asfloat(q1.z), asfloat(q1.w) };
                if (gm < p.M)
                {
                    [ForceUnroll]
                    for (uint c = 0; c < 8u; ++c)
                    {
                        uint gn = cn + c8 + c;
                        if (gn < p.N)
                        {
                            uint idx = gm * p.ldC + gn;
                            float v = p.alpha * vals[c];
                            if (useBias)
                                v += bias[gn];
                            if (useResid)
                                v += R[idx];
                            if (useAccum)
                                v += C[idx];
                            C[idx] = v;
                        }
                    }
                }
            }
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void sml_train_gemm_tc_kk(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { tc_gemm_main<1, 1>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_train_gemm_tc_kn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { tc_gemm_main<1, 0>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_train_gemm_tc_mk(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { tc_gemm_main<0, 1>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_train_gemm_tc_mn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { tc_gemm_main<0, 0>(gid, tid); }

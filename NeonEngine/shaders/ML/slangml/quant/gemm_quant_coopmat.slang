// Prefill GEMM on cooperative matrices (tensor cores) with on-the-fly weight
// dequantization (docs/quant.md).
//
//   C[m][n] = alpha * sum_k A[m][k] * W[n][k]  (+ bias[n]) (+ residual[m][n])
//
// A: fp32 activations [M][lda]. W: Q4 / Q8 / F16 / BF16 rows [N][K] in the
// gemv_quant.slang layouts. Both are converted to fp16 tiles in shared memory;
// the MMA is fp16 x fp16 -> fp32 (16x16x16). K must be a multiple of 32.
// Precision: A is rounded to fp16 (|A| must stay < 65504); accumulation is fp32.
//
// Tile: BM x 128 x 32 per 256-thread workgroup (8 subgroups of 32 lanes).
//   _coopmat      BM = 128, subgroups 4 (M) x 2 (N), 32 x 64 per subgroup
//   _coopmat_m64  BM = 64,  subgroups 2 (M) x 4 (N), 32 x 32 per subgroup
// Pipeline: two shared stages; the next K tile is fetched into registers while
// the current one is multiplied, then written to the other stage (one barrier
// per K tile). Shared memory: 2 * (128 + 128) * 80 bytes = 40 KiB; the epilogue
// reuses the B stages, so two workgroups fit per Ampere SM.
// Requires VK_KHR_cooperative_matrix with fp16 16x16x16 -> fp32 and 32-wide
// subgroups.
//
// Grid: x = ceil(N / 128), y = ceil(M / BM).
// Tensor table: 0 W codes, 1 W meta, 2 A, 3 C, 4 bias, 5 residual.
import slangml_common;
import slangml_llm;

using namespace linalg;

typealias SmlMatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias SmlMatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias SmlMatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

static const uint SML_GEMM_BN = 128u;
static const uint SML_GEMM_BK = 32u;
static const uint SML_GEMM_LDS = 40u;  // padded row stride (halves): 80 bytes, 16-byte aligned
static const uint SML_GEMM_LDS4 = 5u;  // the same stride in uint4 units (8 halves)
static const uint SML_GEMM_FLAG_BIAS = 1u;
static const uint SML_GEMM_FLAG_RESIDUAL = 2u;

struct GemmQuantParams
{
    KernelHeader header;
    uint M;           // 48
    uint N;           // 52
    uint K;           // 56
    uint groupShift;  // 60
    uint lda;         // 64
    uint ldc;         // 68
    uint ldr;         // 72
    uint flags;       // 76
    float alpha;      // 80
    uint reserved0;   // 84
    uint reserved1;   // 88
    uint reserved2;   // 92
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGemmCoopPush;

// fp16 tiles stored as uint4 (8 halves) so every shared store is 128-bit;
// OpCooperativeMatrixLoadKHR takes the stride in uint4 units. The 80-byte row
// stride keeps the 8-row fragment loads bank-conflict free.
groupshared uint4 smlGemmA[2 * 128 * SML_GEMM_LDS4];
groupshared uint4 smlGemmB[2 * SML_GEMM_BN * SML_GEMM_LDS4];

// Raw global data for one K tile, held in registers between fetch and store.
struct GemmStage
{
    float4 a[4];   // 16 consecutive fp32 activations (BM = 128: one row half)
    uint4 b0;      // weight payload (format dependent)
    uint4 b1;
    uint bm;       // Q4/Q8 group meta
};

[ForceInline]
uint sml_pack_half2(float a, float b)
{
    return (f32tof16(a) & 0xFFFFu) | (f32tof16(b) << 16u);
}

[ForceInline]
uint4 sml_pack_half8(float4 lo, float4 hi)
{
    return uint4(sml_pack_half2(lo.x, lo.y), sml_pack_half2(lo.z, lo.w),
                 sml_pack_half2(hi.x, hi.y), sml_pack_half2(hi.z, hi.w));
}

[ForceInline]
void sml_store_half8(uint offset, uint4 v, bool toA)
{
    // offset in halves, multiple of 8
    if (toA)
        smlGemmA[offset >> 3u] = v;
    else
        smlGemmB[offset >> 3u] = v;
}

// Thread t covers A rows t/2 (+128/BM wrap for BM=64: rows t/4, quarter t%4)
// and weight row t/2, k-half t%2.
[ForceInline]
void sml_gemm_fetch<let FMT : int, let BM : int>(
    GemmQuantParams* p, uint tidx, uint m0, uint n0, uint kt,
    uint64_t aBase, uint64_t codes, uint* meta, out GemmStage st)
{
    uint k0 = kt * SML_GEMM_BK;
    // A: BM rows x 32 floats. BM=128: 2 threads/row x 16 floats; BM=64: 4 threads/row x 8 floats.
    if (BM == 128)
    {
        uint row = tidx >> 1u;
        uint kh = (tidx & 1u) * 16u;
        uint gm = m0 + row;
        float4* ap = reinterpret<float4*>(aBase + (uint64_t(min(gm, p.M - 1u)) * p.lda + k0 + kh) * 4u);
        bool ok = gm < p.M;
        [ForceUnroll]
        for (uint i = 0; i < 4; ++i)
            st.a[i] = ok ? sml_ld_f4(ap + i) : float4(0.0f);
    }
    else
    {
        uint row = tidx >> 2u;
        uint kq = (tidx & 3u) * 8u;
        uint gm = m0 + row;
        float4* ap = reinterpret<float4*>(aBase + (uint64_t(min(gm, p.M - 1u)) * p.lda + k0 + kq) * 4u);
        bool ok = gm < p.M;
        st.a[0] = ok ? sml_ld_f4(ap) : float4(0.0f);
        st.a[1] = ok ? sml_ld_f4(ap + 1) : float4(0.0f);
        st.a[2] = float4(0.0f);
        st.a[3] = float4(0.0f);
    }
    // W: 128 rows x 32 k, two threads per row (k halves).
    uint wrow = min(n0 + (tidx >> 1u), p.N - 1u);
    uint khalf = tidx & 1u;
    uint nb = p.K >> 5u;
    if (FMT == SML_WFMT_Q4)
    {
        st.b0 = sml_ld_u4(reinterpret<uint4*>(codes) + (wrow * nb + kt));
        st.b1 = uint4(0u);
        st.bm = meta[wrow * (nb >> p.groupShift) + (kt >> p.groupShift)];
    }
    else if (FMT == SML_WFMT_Q8)
    {
        st.b0 = sml_ld_u4(reinterpret<uint4*>(codes) + ((wrow * nb + kt) * 2u + khalf));
        st.b1 = uint4(0u);
        st.bm = meta[wrow * (nb >> p.groupShift) + (kt >> p.groupShift)];
    }
    else
    {
        uint4* wp = reinterpret<uint4*>(codes) + (wrow * (p.K >> 3u) + kt * 4u + khalf * 2u);
        st.b0 = sml_ld_u4(wp);
        st.b1 = sml_ld_u4(wp + 1);
        st.bm = 0u;
    }
}

[ForceInline]
void sml_gemm_store_stage<let FMT : int, let BM : int>(uint tidx, uint stage, GemmStage st)
{
    if (BM == 128)
    {
        uint row = tidx >> 1u;
        uint kh = (tidx & 1u) * 16u;
        uint base = stage * 128u * SML_GEMM_LDS + row * SML_GEMM_LDS + kh;
        sml_store_half8(base, sml_pack_half8(st.a[0], st.a[1]), true);
        sml_store_half8(base + 8u, sml_pack_half8(st.a[2], st.a[3]), true);
    }
    else
    {
        uint row = tidx >> 2u;
        uint kq = (tidx & 3u) * 8u;
        uint base = stage * 128u * SML_GEMM_LDS + row * SML_GEMM_LDS + kq;
        sml_store_half8(base, sml_pack_half8(st.a[0], st.a[1]), true);
    }
    uint wrow = tidx >> 1u;
    uint khalf = tidx & 1u;
    uint bbase = stage * SML_GEMM_BN * SML_GEMM_LDS + wrow * SML_GEMM_LDS + khalf * 16u;
    if (FMT == SML_WFMT_Q4)
    {
        // khalf 0 -> elements 0..15 (low nibbles), 1 -> 16..31 (high nibbles).
        float2 dm = sml_half2_to_float2(st.bm);
        uint shift = khalf * 4u;
        float4 v[4];
        [ForceUnroll]
        for (uint j = 0; j < 4; ++j)
        {
            uint word = (j == 0u ? st.b0.x : (j == 1u ? st.b0.y : (j == 2u ? st.b0.z : st.b0.w))) >> shift;
            v[j] = float4(float(word & 0xFu), float((word >> 8u) & 0xFu),
                          float((word >> 16u) & 0xFu), float((word >> 24u) & 0xFu)) * dm.x + dm.y;
        }
        sml_store_half8(bbase, sml_pack_half8(v[0], v[1]), false);
        sml_store_half8(bbase + 8u, sml_pack_half8(v[2], v[3]), false);
    }
    else if (FMT == SML_WFMT_Q8)
    {
        float2 dm = sml_half2_to_float2(st.bm);
        float4 v0 = sml_q8_word_to_float(st.b0.x, dm);
        float4 v1 = sml_q8_word_to_float(st.b0.y, dm);
        float4 v2 = sml_q8_word_to_float(st.b0.z, dm);
        float4 v3 = sml_q8_word_to_float(st.b0.w, dm);
        sml_store_half8(bbase, sml_pack_half8(v0, v1), false);
        sml_store_half8(bbase + 8u, sml_pack_half8(v2, v3), false);
    }
    else if (FMT == SML_WFMT_F16)
    {
        sml_store_half8(bbase, st.b0, false);
        sml_store_half8(bbase + 8u, st.b1, false);
    }
    else
    {
        float4 lo, hi;
        sml_unpack_bf16x8(st.b0, lo, hi);
        sml_store_half8(bbase, sml_pack_half8(lo, hi), false);
        sml_unpack_bf16x8(st.b1, lo, hi);
        sml_store_half8(bbase + 8u, sml_pack_half8(lo, hi), false);
    }
}

[ForceInline]
void sml_gemm_coop_main<let FMT : int, let BM : int, let WARPS_M : int>(uint3 gid, uint3 tid)
{
    static const int WARPS_N = 8 / WARPS_M;
    static const int FM = BM / WARPS_M / 16;    // 16x16 fragments per subgroup along M
    static const int FN = 128 / WARPS_N / 16;   // ... along N
    GemmQuantParams* p = reinterpret<GemmQuantParams*>(smlGemmCoopPush.headerAddress);
    uint tidx = tid.x;
    uint warp = tidx >> 5u;
    uint lane = tidx & 31u;
    uint wm = warp / uint(WARPS_N);
    uint wn = warp % uint(WARPS_N);
    uint n0 = (p.header.dispatchBase.x + gid.x) * SML_GEMM_BN;
    uint m0 = (p.header.dispatchBase.y + gid.y) * uint(BM);

    uint64_t A = sml_base(slangml_tensor_desc(p.header, 2u), 4u);
    uint64_t codes = sml_base(slangml_tensor_desc(p.header, 0u), 1u);
    uint* meta = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u));

    SmlMatC acc[FM][FN];
    [ForceUnroll]
    for (int i = 0; i < FM; ++i)
        [ForceUnroll]
        for (int j = 0; j < FN; ++j)
            acc[i][j] = SmlMatC(0.0f);

    uint nk = p.K / SML_GEMM_BK;
    GemmStage st;
    sml_gemm_fetch<FMT, BM>(p, tidx, m0, n0, 0u, A, codes, meta, st);
    sml_gemm_store_stage<FMT, BM>(tidx, 0u, st);
    GroupMemoryBarrierWithGroupSync();

    for (uint kt = 0; kt < nk; ++kt)
    {
        uint cur = kt & 1u;
        bool more = kt + 1u < nk;
        if (more)
            sml_gemm_fetch<FMT, BM>(p, tidx, m0, n0, kt + 1u, A, codes, meta, st);
        [ForceUnroll]
        for (uint ks = 0; ks < SML_GEMM_BK / 16u; ++ks)
        {
            SmlMatA fa[FM];
            SmlMatB fb[FN];
            [ForceUnroll]
            for (int i = 0; i < FM; ++i)
                fa[i] = SmlMatA.Load<CoopMatMatrixLayout.RowMajor>(
                    smlGemmA, cur * 128u * SML_GEMM_LDS4 + (wm * uint(FM * 16) + uint(i) * 16u) * SML_GEMM_LDS4 + ks * 2u,
                    SML_GEMM_LDS4);
            [ForceUnroll]
            for (int j = 0; j < FN; ++j)
                fb[j] = SmlMatB.Load<CoopMatMatrixLayout.ColumnMajor>(
                    smlGemmB, cur * SML_GEMM_BN * SML_GEMM_LDS4 + (wn * uint(FN * 16) + uint(j) * 16u) * SML_GEMM_LDS4 + ks * 2u,
                    SML_GEMM_LDS4);
            [ForceUnroll]
            for (int i = 0; i < FM; ++i)
                [ForceUnroll]
                for (int j = 0; j < FN; ++j)
                    acc[i][j] = coopMatMulAdd<float, false>(fa[i], fb[j], acc[i][j]);
        }
        if (more)
            sml_gemm_store_stage<FMT, BM>(tidx, cur ^ 1u, st);
        GroupMemoryBarrierWithGroupSync();
    }

    // Epilogue.
    uint64_t cBase = sml_base(slangml_tensor_desc(p.header, 3u), 4u);
    float* bias = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u));
    uint64_t resBase = sml_base(slangml_tensor_desc(p.header, 5u), 4u);
    bool plain = p.flags == 0u && p.alpha == 1.0f && m0 + uint(BM) <= p.M && n0 + SML_GEMM_BN <= p.N;
    [ForceUnroll]
    for (int i = 0; i < FM; ++i)
    {
        [ForceUnroll]
        for (int j = 0; j < FN; ++j)
        {
            uint rm = m0 + wm * uint(FM * 16) + uint(i) * 16u;
            uint cn = n0 + wn * uint(FN * 16) + uint(j) * 16u;
            if (plain)
            {
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(
                    reinterpret<float*>(cBase + (uint64_t(rm) * p.ldc + cn) * 4u), 0u, p.ldc);
            }
            else
            {
                // Per-warp 16x16 fp32 scratch in the (now idle) B stages: 64 uint4, stride 4.
                acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(smlGemmB, warp * 64u, 4u);
                // Lane l handles row l/2, columns (l%2)*8 .. +8 of the 16x16 fragment.
                uint r = lane >> 1u;
                uint c0 = (lane & 1u) * 8u;
                uint gm = rm + r;
                uint4 q0 = smlGemmB[warp * 64u + r * 4u + (c0 >> 2u)];
                uint4 q1 = smlGemmB[warp * 64u + r * 4u + (c0 >> 2u) + 1u];
                float vals[8] = { asfloat(q0.x), asfloat(q0.y), asfloat(q0.z), asfloat(q0.w),
                                  asfloat(q1.x), asfloat(q1.y), asfloat(q1.z), asfloat(q1.w) };
                [ForceUnroll]
                for (uint c = 0; c < 8; ++c)
                {
                    uint gn = cn + c0 + c;
                    float v = vals[c] * p.alpha;
                    if (gm < p.M && gn < p.N)
                    {
                        if ((p.flags & SML_GEMM_FLAG_BIAS) != 0u)
                            v += bias[gn];
                        if ((p.flags & SML_GEMM_FLAG_RESIDUAL) != 0u)
                            v += reinterpret<float*>(resBase + (uint64_t(gm) * p.ldr + gn) * 4u)[0];
                        reinterpret<float*>(cBase + (uint64_t(gm) * p.ldc + gn) * 4u)[0] = v;
                    }
                }
            }
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_q4_coopmat(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_Q4, 128, 4>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_q8_coopmat(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_Q8, 128, 4>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_f16_coopmat(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_F16, 128, 4>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_bf16_coopmat(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_BF16, 128, 4>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_q4_coopmat_m64(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_Q4, 64, 2>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_q8_coopmat_m64(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_Q8, 64, 2>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_gemm_fwd_bf16_coopmat_m64(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemm_coop_main<SML_WFMT_BF16, 64, 2>(gid, tid); }

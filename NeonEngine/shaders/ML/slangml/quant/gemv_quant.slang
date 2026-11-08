// Decode-path matrix-vector products with quantized or 16-bit weights
// (docs/quant.md). Computes, for every activation row m < M and output n < N,
//
//   y[m][n] = alpha * sum_k x[m][k] * W[n][k]  (+ bias[n]) (+ residual[m][n])
//
// and the fused gate/up form for gated MLPs:
//
//   y[m][n] = glu(alpha * x.Wg[n] + bg[n], alpha * x.Wu[n] + bu[n])
//
// Weight layouts (K multiple of 32, row n contiguous):
//   Q4   codes  uint4 per 32-block [N][K/32]; meta half2 (d, m) [N][K/group]
//   Q8   codes  2 x uint4 per block [N][K/32]; meta as Q4
//   F16 / BF16  [N][K]
//   MXFP4  codes as Q4 (split-nibble e2m1); meta one e8m0 byte per block [N][K/32]
// Q4/Q8/MXFP4 consume int8 activations from sml_quant_quantize_q8act_fwd_f32
// (Xq int8 [M][K], Xs float2 [M][K/32]); F16/BF16 consume fp32 X [M][K].
//
// Mapping: 128-thread workgroup; each subgroup owns ROWS consecutive
// output rows and strides its lanes across K (one 32-block per lane per step),
// so a subgroup reads ROWS fully coalesced weight rows and shares every
// activation load across them. The activation rows (MC) stay in registers.
// No shared memory and no workgroup barriers.
//
// Tensor table: 0 W codes, 1 W meta, 2 Xq | X, 3 Xs, 4 Y, 5 bias (gate bias),
//               6 residual (up bias for _swiglu), 7 Wu codes, 8 Wu meta.
// Dispatch: x = ceil(N / (ROWS * 128 / subgroupSize)). ROWS per variant:
//   _m1 -> 2, _m2 -> 4, _m4 -> 4, _m8 -> 8 (so 8/16/16/32 rows per workgroup on
//   32-wide subgroups). Measured on GA107: more rows amortise the per-row
//   activation loads, which dominate once M > 1. -D SML_GEMV_ROWS=<n> forces
//   one value for every variant (tuning only).
import slangml_common;
import slangml_llm;

#ifdef SML_GEMV_ROWS
#define SML_ROWS_M1 SML_GEMV_ROWS
#define SML_ROWS_M2 SML_GEMV_ROWS
#define SML_ROWS_M4 SML_GEMV_ROWS
#define SML_ROWS_M8 SML_GEMV_ROWS
#else
#define SML_ROWS_M1 2
#define SML_ROWS_M2 4
#define SML_ROWS_M4 4
#define SML_ROWS_M8 8
#endif

static const uint SML_GEMV_WG = 128u;
static const uint SML_GEMV_FLAG_BIAS = 1u;
static const uint SML_GEMV_FLAG_RESIDUAL = 2u;

struct GemvParams
{
    KernelHeader header;
    uint N;           // 48  output rows
    uint K;           // 52  input dim (multiple of 32)
    uint M;           // 56  activation rows, 1 <= M <= MC of the variant
    uint groupShift;  // 60  log2(group / 32) for the meta index
    uint flags;       // 64  SML_GEMV_FLAG_*
    float alpha;      // 68
    uint ldY;         // 72  output row stride (elements)
    uint ldR;         // 76  residual row stride (elements)
    uint actMode;     // 80  SML_ACT_* for _swiglu
    float actLimit;   // 84  clamp for SML_ACT_SWIGLU_OAI
    float actAlpha;   // 88  sigmoid slope for SML_ACT_SWIGLU_OAI
    uint reserved;    // 92
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGemvPush;

#include "gemv_core.slang"

[ForceInline]
void sml_gemv_accumulate<let FMT : int, let MC : int, let ROWS : int>(
    GemvParams* p, uint codesIndex, uint metaIndex, uint rowBase, uint lane, uint laneCount,
    inout GemvAcc<MC, ROWS> acc)
{
    TensorDesc wd = slangml_tensor_desc(p.header, codesIndex);
    TensorDesc xd = slangml_tensor_desc(p.header, 2u);
    GemvShape shape = { p.N, p.K, p.M, p.groupShift };
    if (FMT == SML_WFMT_Q4 || FMT == SML_WFMT_Q8 || FMT == SML_WFMT_MXFP4)
    {
        TensorDesc md = slangml_tensor_desc(p.header, metaIndex);
        TensorDesc sd = slangml_tensor_desc(p.header, 3u);
        sml_gemv_quant_rows<FMT, MC, ROWS>(shape, sml_base(wd, 1u), sml_base(md, FMT == SML_WFMT_MXFP4 ? 1u : 4u), sml_base(xd, 1u),
                                     sml_base(sd, 8u), rowBase, lane, laneCount, acc);
    }
    else
    {
        sml_gemv_f16_rows<FMT, MC, ROWS>(shape, sml_base(wd, 2u), sml_base(xd, 4u), rowBase, lane, laneCount, acc);
    }
}

[ForceInline]
void sml_gemv_main<let FMT : int, let MC : int, let ROWS : int>(uint3 gid, uint3 tid)
{
    GemvParams* p = reinterpret<GemvParams*>(smlGemvPush.headerAddress);
    uint lane = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint subgroup = tid.x / laneCount;
    uint subgroups = SML_GEMV_WG / laneCount;
    uint rowBase = ((p.header.dispatchBase.x + gid.x) * subgroups + subgroup) * ROWS;
    if (rowBase >= p.N)
        return;  // uniform per subgroup; the kernel has no workgroup barriers

    GemvAcc<MC, ROWS> acc;
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
            acc.v[r][m] = 0.0f;
    sml_gemv_accumulate<FMT, MC, ROWS>(p, 0u, 1u, rowBase, lane, laneCount, acc);

    TensorDesc yd = slangml_tensor_desc(p.header, 4u);
    float* y = reinterpret<float*>(sml_base(yd, 4u));
    bool hasBias = (p.flags & SML_GEMV_FLAG_BIAS) != 0u;
    bool hasResidual = (p.flags & SML_GEMV_FLAG_RESIDUAL) != 0u;
    // Optional descriptors are only dereferenced when their flag is set.
    float* bias = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float* residual = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u));
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
    {
        uint row = rowBase + r;
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
        {
            float sum = WaveActiveSum(acc.v[r][m]);
            if (lane == 0u && row < p.N && m < p.M)
            {
                float v = p.alpha * sum;
                if (hasBias)
                    v += bias[row];
                if (hasResidual)
                    v += residual[m * p.ldR + row];
                y[m * p.ldY + row] = v;
            }
        }
    }
}

[ForceInline]
void sml_gemv_swiglu_main<let FMT : int, let MC : int, let ROWS : int>(uint3 gid, uint3 tid)
{
    GemvParams* p = reinterpret<GemvParams*>(smlGemvPush.headerAddress);
    uint lane = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint subgroup = tid.x / laneCount;
    uint subgroups = SML_GEMV_WG / laneCount;
    uint rowBase = ((p.header.dispatchBase.x + gid.x) * subgroups + subgroup) * ROWS;
    if (rowBase >= p.N)
        return;

    GemvAcc<MC, ROWS> accG;
    GemvAcc<MC, ROWS> accU;
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
        {
            accG.v[r][m] = 0.0f;
            accU.v[r][m] = 0.0f;
        }
    sml_gemv_accumulate<FMT, MC, ROWS>(p, 0u, 1u, rowBase, lane, laneCount, accG);
    sml_gemv_accumulate<FMT, MC, ROWS>(p, 7u, 8u, rowBase, lane, laneCount, accU);

    TensorDesc yd = slangml_tensor_desc(p.header, 4u);
    float* y = reinterpret<float*>(sml_base(yd, 4u));
    bool hasBias = (p.flags & SML_GEMV_FLAG_BIAS) != 0u;
    float* biasG = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float* biasU = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u));
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
    {
        uint row = rowBase + r;
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
        {
            float g = WaveActiveSum(accG.v[r][m]) * p.alpha;
            float u = WaveActiveSum(accU.v[r][m]) * p.alpha;
            if (lane == 0u && row < p.N && m < p.M)
            {
                if (hasBias)
                {
                    g += biasG[row];
                    u += biasU[row];
                }
                y[m * p.ldY + row] = sml_glu(p.actMode, g, u, p.actLimit, p.actAlpha);
            }
        }
    }
}

// --- plain projections ------------------------------------------------------
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q4_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q4, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q4_m2(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q4, 2, SML_ROWS_M2>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q4_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q4, 4, SML_ROWS_M4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q4_m8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q4, 8, SML_ROWS_M8>(gid, tid); }

[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q8_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q8, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q8_m2(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q8, 2, SML_ROWS_M2>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q8_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q8, 4, SML_ROWS_M4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_q8_m8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_Q8, 8, SML_ROWS_M8>(gid, tid); }

[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_mxfp4_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_MXFP4, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_mxfp4_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_MXFP4, 4, SML_ROWS_M4>(gid, tid); }

[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_f16_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_F16, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_f16_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_F16, 4, SML_ROWS_M4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_bf16_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_BF16, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_fwd_bf16_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_main<SML_WFMT_BF16, 4, SML_ROWS_M4>(gid, tid); }

// --- fused gate/up (gated MLP) ----------------------------------------------
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_q4_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_Q4, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_q4_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_Q4, 4, SML_ROWS_M4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_q8_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_Q8, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_q8_m4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_Q8, 4, SML_ROWS_M4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_f16_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_F16, 1, SML_ROWS_M1>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_swiglu_fwd_bf16_m1(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_gemv_swiglu_main<SML_WFMT_BF16, 1, SML_ROWS_M1>(gid, tid); }

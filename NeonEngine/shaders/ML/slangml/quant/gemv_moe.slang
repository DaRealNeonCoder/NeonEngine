// Mixture-of-experts decode GEMVs (docs/quant.md). Expert weights are stacked
// [E][N][K] in the same formats as gemv_quant.slang; `codeStride`/`metaStride`
// are the bytes between experts. A "slot" is one (token, selected expert) pair
// from sml_llm_moe_router_fwd_f32, slot s belongs to token s / slotsPerToken.
//
// _swiglu: y[s][n] = glu(alpha * x[tok(s)] . Wg[e_s][n] + bg[e_s][n],
//                        alpha * x[tok(s)] . Wu[e_s][n] + bu[e_s][n])
//          grid: x = row tiles over N, y = numSlots.
// _down:   y[tok][n] = sum_{s of tok} w_s * (alpha * x[s] . Wd[e_s][n] + bd[e_s][n])
//                      (+ residual[tok][n])
//          grid: x = row tiles over N, y = tokens. The weighted sum over the
//          token's experts happens in registers before the subgroup reduction.
// Row tiles and ROWS per subgroup match the _m1 GEMV (2 rows per subgroup).
//
// Tensor table: 0 W codes, 1 W meta, 2 Xq | X, 3 Xs, 4 Y, 5 bias (gate bias),
//   6 residual (down) | up bias (swiglu), 7 Wu codes, 8 Wu meta,
//   9 expert ids i32 [numSlots], 10 routing weights fp32 [numSlots],
//   11 expert address table (SML_MOE_FLAG_EXPERT_TABLE).
//
// Expert address table (flag 4): uint64 [E][4] = {codes, meta, codes2, meta2}
// byte addresses per expert; entries 0/1 are W (gate for _swiglu, down for
// _down), 2/3 are Wu (_swiglu only). Tensors 0/1/7/8 and codeStride/metaStride
// are then unused. This lets each expert live anywhere the GPU can address:
// a VRAM cache slot or host-visible system memory (read over PCIe), which the
// host can change between tokens without touching the kernels.
import slangml_common;
import slangml_llm;

#include "gemv_core.slang"

static const uint SML_MOE_ROWS = 2u;
static const uint SML_MOE_WG = 128u;
static const uint SML_MOE_FLAG_BIAS = 1u;
static const uint SML_MOE_FLAG_RESIDUAL = 2u;
static const uint SML_MOE_FLAG_EXPERT_TABLE = 4u;

struct MoeGemvParams
{
    KernelHeader header;
    uint N;              // 48
    uint K;              // 52
    uint numSlots;       // 56
    uint groupShift;     // 60
    uint flags;          // 64
    float alpha;         // 68
    uint ldY;            // 72
    uint actMode;        // 76
    float actLimit;      // 80
    float actAlpha;      // 84
    uint codeStride;     // 88 bytes between experts (codes)
    uint metaStride;     // 92 bytes between experts (meta; unused for f16/bf16)
    uint biasStride;     // 96 elements between experts (bias)
    uint slotsPerToken;  // 100
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlMoePush;

// Accumulate one slot's dot products against expert `expert` for activation row `xRow`.
[ForceInline]
void sml_moe_accumulate<let FMT : int>(
    MoeGemvParams* p, uint codesIndex, uint metaIndex, uint expert, uint xRow,
    uint rowBase, uint lane, uint laneCount, inout GemvAcc<1, SML_MOE_ROWS> acc)
{
    GemvShape shape = { p.N, p.K, 1u, p.groupShift };
    uint64_t codes;
    uint64_t meta;
    if ((p.flags & SML_MOE_FLAG_EXPERT_TABLE) != 0u)
    {
        uint64_t* table = reinterpret<uint64_t*>(sml_base(slangml_tensor_desc(p.header, 11u), 8u));
        uint entry = expert * 4u + (codesIndex == 0u ? 0u : 2u);
        codes = table[entry];
        meta = table[entry + 1u];
    }
    else
    {
        codes = sml_base(slangml_tensor_desc(p.header, codesIndex), 1u) + uint64_t(expert) * p.codeStride;
        meta = sml_base(slangml_tensor_desc(p.header, metaIndex), FMT == SML_WFMT_MXFP4 ? 1u : 4u) + uint64_t(expert) * p.metaStride;
    }
    TensorDesc xd = slangml_tensor_desc(p.header, 2u);
    if (FMT == SML_WFMT_Q4 || FMT == SML_WFMT_Q8 || FMT == SML_WFMT_MXFP4)
    {
        uint64_t xq = sml_base(xd, 1u) + uint64_t(xRow) * p.K;
        uint64_t xs = sml_base(slangml_tensor_desc(p.header, 3u), 8u) + uint64_t(xRow) * (p.K >> 5u) * 8u;
        sml_gemv_quant_rows<FMT, 1, SML_MOE_ROWS>(shape, codes, meta, xq, xs, rowBase, lane, laneCount, acc);
    }
    else
    {
        uint64_t x = sml_base(xd, 4u) + uint64_t(xRow) * p.K * 4u;
        sml_gemv_f16_rows<FMT, 1, SML_MOE_ROWS>(shape, codes, x, rowBase, lane, laneCount, acc);
    }
}

[ForceInline]
void sml_moe_swiglu_main<let FMT : int>(uint3 gid, uint3 tid)
{
    MoeGemvParams* p = reinterpret<MoeGemvParams*>(smlMoePush.headerAddress);
    uint lane = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint subgroups = SML_MOE_WG / laneCount;
    uint rowBase = ((p.header.dispatchBase.x + gid.x) * subgroups + tid.x / laneCount) * SML_MOE_ROWS;
    uint slot = gid.y;
    if (rowBase >= p.N || slot >= p.numSlots)
        return;
    uint expert = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 9u), 4u))[slot];
    uint xRow = slot / p.slotsPerToken;

    GemvAcc<1, SML_MOE_ROWS> accG, accU;
    [ForceUnroll]
    for (uint r = 0; r < SML_MOE_ROWS; ++r)
    {
        accG.v[r][0] = 0.0f;
        accU.v[r][0] = 0.0f;
    }
    sml_moe_accumulate<FMT>(p, 0u, 1u, expert, xRow, rowBase, lane, laneCount, accG);
    sml_moe_accumulate<FMT>(p, 7u, 8u, expert, xRow, rowBase, lane, laneCount, accU);

    float* y = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u)) + slot * p.ldY;
    float* bg = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u)) + expert * p.biasStride;
    float* bu = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u)) + expert * p.biasStride;
    bool hasBias = (p.flags & SML_MOE_FLAG_BIAS) != 0u;
    [ForceUnroll]
    for (uint r = 0; r < SML_MOE_ROWS; ++r)
    {
        uint row = rowBase + r;
        float g = WaveActiveSum(accG.v[r][0]) * p.alpha;
        float u = WaveActiveSum(accU.v[r][0]) * p.alpha;
        if (lane == 0u && row < p.N)
        {
            if (hasBias)
            {
                g += bg[row];
                u += bu[row];
            }
            y[row] = sml_glu(p.actMode, g, u, p.actLimit, p.actAlpha);
        }
    }
}

[ForceInline]
void sml_moe_down_main<let FMT : int>(uint3 gid, uint3 tid)
{
    MoeGemvParams* p = reinterpret<MoeGemvParams*>(smlMoePush.headerAddress);
    uint lane = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint subgroups = SML_MOE_WG / laneCount;
    uint rowBase = ((p.header.dispatchBase.x + gid.x) * subgroups + tid.x / laneCount) * SML_MOE_ROWS;
    uint token = gid.y;
    if (rowBase >= p.N)
        return;
    uint* ids = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 9u), 4u));
    float* weights = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 10u), 4u));
    float* bias = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    bool hasBias = (p.flags & SML_MOE_FLAG_BIAS) != 0u;

    float total[SML_MOE_ROWS];
    float biasSum[SML_MOE_ROWS];
    [ForceUnroll]
    for (uint r = 0; r < SML_MOE_ROWS; ++r)
    {
        total[r] = 0.0f;
        biasSum[r] = 0.0f;
    }
    for (uint s = token * p.slotsPerToken; s < (token + 1u) * p.slotsPerToken; ++s)
    {
        uint expert = ids[s];
        float wgt = weights[s];
        GemvAcc<1, SML_MOE_ROWS> acc;
        [ForceUnroll]
        for (uint r = 0; r < SML_MOE_ROWS; ++r)
            acc.v[r][0] = 0.0f;
        sml_moe_accumulate<FMT>(p, 0u, 1u, expert, s, rowBase, lane, laneCount, acc);
        [ForceUnroll]
        for (uint r = 0; r < SML_MOE_ROWS; ++r)
        {
            total[r] = fma(wgt, acc.v[r][0], total[r]);
            if (hasBias)
                biasSum[r] = fma(wgt, bias[expert * p.biasStride + min(rowBase + r, p.N - 1u)], biasSum[r]);
        }
    }

    float* y = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u)) + token * p.ldY;
    float* residual = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 6u), 4u)) + token * p.ldY;
    [ForceUnroll]
    for (uint r = 0; r < SML_MOE_ROWS; ++r)
    {
        uint row = rowBase + r;
        float v = WaveActiveSum(total[r]) * p.alpha + biasSum[r];
        if (lane == 0u && row < p.N)
        {
            if ((p.flags & SML_MOE_FLAG_RESIDUAL) != 0u)
                v += residual[row];
            y[row] = v;
        }
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_swiglu_fwd_q4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_swiglu_main<SML_WFMT_Q4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_swiglu_fwd_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_swiglu_main<SML_WFMT_Q8>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_swiglu_fwd_bf16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_swiglu_main<SML_WFMT_BF16>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_swiglu_fwd_mxfp4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_swiglu_main<SML_WFMT_MXFP4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_down_fwd_q4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_down_main<SML_WFMT_Q4>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_down_fwd_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_down_main<SML_WFMT_Q8>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_down_fwd_bf16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_down_main<SML_WFMT_BF16>(gid, tid); }
[shader("compute")] [numthreads(128, 1, 1)]
void sml_quant_gemv_moe_down_fwd_mxfp4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_moe_down_main<SML_WFMT_MXFP4>(gid, tid); }

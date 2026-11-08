// Fused RMSNorm for decoder blocks (docs/llm_ops.md).
//
//   h = x (+ r)                                   SML_NORM_ADD_RESIDUAL
//   H = h                                         SML_NORM_WRITE_SUM (may alias x or r)
//   y = h * rsqrt(mean(h^2) + eps) * w'           w' = (1 + w) with SML_NORM_UNIT_OFFSET
//   y = y * silu(z)                               SML_NORM_GATE_SILU (Qwen3.5 linear-attn output)
//   Y = y (fp32)                                  SML_NORM_OUT_F32
//   Yq, Ys = int8 blocks of y                     SML_NORM_OUT_Q8 (same format as quantize_f32)
//
// All math is fp32. cols must be a multiple of 32 (the int8 block size).
// Each thread owns float4 quads q = lane + lanes * i, so a 32-element int8
// block is 8 consecutive lanes and the row is held in registers between the
// statistics and the output pass (h is read once even when H aliases x/r).
//
// _row:   one 256-thread workgroup per row, cols <= 256 * 4 * SML_NORM_MAX_QUADS.
// _small: one 32-lane group per row (4 rows per 128-thread workgroup),
//         cols <= 32 * 4 * SML_NORM_MAX_QUADS. For head-sized rows (64..1024).
// Dispatch: _row x = rows; _small x = ceil(rows / 4).
//
// Tensor table: 0 x, 1 r, 2 w, 3 Y, 4 H, 5 z, 6 Yq (int8), 7 Ys (float2).
import slangml_common;
import slangml_llm;

#ifndef SML_NORM_MAX_QUADS
#define SML_NORM_MAX_QUADS 8
#endif

static const uint SML_NORM_ADD_RESIDUAL = 1u;
static const uint SML_NORM_WRITE_SUM = 2u;
static const uint SML_NORM_UNIT_OFFSET = 4u;
static const uint SML_NORM_GATE_SILU = 8u;
static const uint SML_NORM_OUT_F32 = 16u;
static const uint SML_NORM_OUT_Q8 = 32u;

struct RMSNormFusedParams
{
    KernelHeader header;
    uint rows;    // 48
    uint cols;    // 52
    float eps;    // 56
    uint flags;   // 60
    uint ldX;     // 64  row strides in elements
    uint ldR;     // 68
    uint ldY;     // 72  (also used for H)
    uint ldZ;     // 76
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlNormPush;

groupshared float smlNormPartial[8];

// lanesPerRow is 256 (_row) or 32 (_small). `rowLane` is the thread index in
// its row, `lane32Base` the first physical lane of its 32-lane group.
[ForceInline]
void sml_rmsnorm_row(RMSNormFusedParams* p, uint row, uint rowLane, uint lanesPerRow, bool active, uint tidx)
{
    uint quads = p.cols >> 2u;
    uint safeRow = min(row, p.rows - 1u);
    float4* x = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u) + uint64_t(safeRow * p.ldX) * 4u);
    float4* r = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u) + uint64_t(safeRow * p.ldR) * 4u);
    bool addResidual = (p.flags & SML_NORM_ADD_RESIDUAL) != 0u;

    float4 hv[SML_NORM_MAX_QUADS];
    float sq = 0.0f;
    [ForceUnroll]
    for (uint i = 0; i < SML_NORM_MAX_QUADS; ++i)
    {
        uint q = rowLane + i * lanesPerRow;
        float4 v = float4(0.0f);
        if (q < quads)
        {
            v = sml_ld_f4(x + q);
            if (addResidual)
                v += sml_ld_f4(r + q);
        }
        hv[i] = v;
        sq += dot(v, v);
    }

    float total;
    if (lanesPerRow == 32u)
    {
        total = sml_cluster_sum(sq, 32u);
    }
    else
    {
        float s = WaveActiveSum(sq);
        uint laneCount = WaveGetLaneCount();
        if (WaveGetLaneIndex() == 0u)
            smlNormPartial[tidx / laneCount] = s;
        GroupMemoryBarrierWithGroupSync();
        total = 0.0f;
        for (uint i = 0; i < lanesPerRow / laneCount; ++i)
            total += smlNormPartial[i];
    }
    float inv = rsqrt(total / float(p.cols) + p.eps);
    if (!active)
        return;  // inactive rows only exist in _small, after its last shuffle

    float4* w = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u));
    float4* yOut = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u) + uint64_t(row * p.ldY) * 4u);
    float4* hOut = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u) + uint64_t(row * p.ldY) * 4u);
    float4* z = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u) + uint64_t(row * p.ldZ) * 4u);
    uint* yq = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 6u), 1u) + uint64_t(row * p.cols));
    float2* ys = reinterpret<float2*>(sml_base(slangml_tensor_desc(p.header, 7u), 8u) + uint64_t(row * (p.cols >> 5u)) * 8u);
    float wOffset = (p.flags & SML_NORM_UNIT_OFFSET) != 0u ? 1.0f : 0.0f;

    [ForceUnroll]
    for (uint i = 0; i < SML_NORM_MAX_QUADS; ++i)
    {
        if (i * lanesPerRow < quads)  // uniform: some lane of the row still has work
        {
            uint q = rowLane + i * lanesPerRow;
            bool live = q < quads;
            uint sq4 = min(q, quads - 1u);
            float4 y = hv[i] * inv * (sml_ld_f4(w + sq4) + wOffset);
            if ((p.flags & SML_NORM_GATE_SILU) != 0u)
            {
                float4 g = sml_ld_f4(z + sq4);
                y *= float4(sml_silu(g.x), sml_silu(g.y), sml_silu(g.z), sml_silu(g.w));
            }
            if (live && (p.flags & SML_NORM_WRITE_SUM) != 0u)
                sml_st_f4(hOut + q, hv[i]);
            if (live && (p.flags & SML_NORM_OUT_F32) != 0u)
                sml_st_f4(yOut + q, y);
            if ((p.flags & SML_NORM_OUT_Q8) != 0u)
            {
                // quads is a multiple of 8, so an 8-lane block is all live or all dead.
                float amax = max(max(abs(y.x), abs(y.y)), max(abs(y.z), abs(y.w)));
                amax = sml_cluster_max(amax, 8u);
                float inv127 = amax > 0.0f ? 127.0f / amax : 0.0f;
                int4 qi = clamp(int4(round(y * inv127)), int4(-127), int4(127));
                float d = amax * (1.0f / 127.0f);
                float qsum = sml_cluster_sum(float(qi.x + qi.y + qi.z + qi.w), 8u);
                if (live)
                {
                    yq[q] = sml_pack_i8x4(qi);
                    if ((q & 7u) == 0u)
                        ys[q >> 3u] = float2(d, d * qsum);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_rmsnorm_fwd_f32_row(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RMSNormFusedParams* p = reinterpret<RMSNormFusedParams*>(smlNormPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;  // uniform per workgroup
    sml_rmsnorm_row(p, row, tid.x, 256u, true, tid.x);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_llm_rmsnorm_fwd_f32_small(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RMSNormFusedParams* p = reinterpret<RMSNormFusedParams*>(smlNormPush.headerAddress);
    uint row = (p.header.dispatchBase.x + gid.x) * 4u + (tid.x >> 5u);
    // Rows past the end keep running (on a clamped row) so 64-wide subgroups
    // never shuffle with an exited half; they skip every store.
    sml_rmsnorm_row(p, row, tid.x & 31u, 32u, row < p.rows, tid.x);
}

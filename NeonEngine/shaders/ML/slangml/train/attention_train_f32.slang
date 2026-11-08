// Causal GQA attention with gpt-oss sinks and an optional sliding window, for
// training (docs/gpt_oss_training.md). head_dim is fixed at 64.
//
// Per (batch b, query head h, query i), kv head = h / (Hq / Hkv):
//   s_j = scale * q_i . k_j      for keys j <= i with (window == 0 or i - j < window)
//   L_i = log(exp(sink_h) + sum_j exp(s_j))       (the sink adds to the denominator only)
//   o_i = sum_j exp(s_j - L_i) v_j
// Backward, with D_i = dO_i . o_i and p_ij = exp(s_ij - L_i):
//   dv_j += p_ij dO_i;  ds_ij = p_ij (dO_i . v_j - D_i)
//   dq_i += scale ds_ij k_j;  dk_j += scale ds_ij q_i;  dsink_h -= sum_i exp(sink_h - L_i) D_i
//
// q, k, v are read from the packed qkv activation [rows][ldQKV] (q heads,
// then k heads, then v heads; row = b * T + position), already rotated.
// T must be a multiple of 64.
//
// Thread layout: 256 threads = 64 rows x 4 lanes. Lane l of a row owns the
// 16 dims {16 j + 4 l + c}; dot products are finished with a 4-lane shuffle sum.
// Keys (or queries) are staged through shared memory 32 at a time.
import slangml_common;
import slangml_llm;
import slangml_train;

static const uint HD = 64u;
static const uint TILE = 32u;

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
ConstantBuffer<KernelPushConstants> smlTAttnPush;

groupshared float smlTaX[TILE * HD];
groupshared float smlTaY[TILE * HD];
groupshared float smlTaL[TILE];
groupshared float smlTaD[TILE];

[ForceInline]
uint sml_ta_dim(uint lane4, uint j) { return 16u * j + 4u * lane4; }

[ForceInline]
void sml_ta_load16(float* src, uint lane4, out float4 v[4])
{
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
        v[j] = sml_ld_f4(reinterpret<float4*>(src + sml_ta_dim(lane4, j)));
}

[ForceInline]
void sml_ta_store16(float* dst, uint lane4, float4 v[4])
{
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
        sml_st_f4(reinterpret<float4*>(dst + sml_ta_dim(lane4, j)), v[j]);
}

// Partial dot of the lane's 16 dims with shared row `r` of `tile`.
[ForceInline]
float sml_ta_dot_x(float4 a[4], uint r, uint lane4)
{
    float s = 0.0f;
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
    {
        uint o = r * HD + sml_ta_dim(lane4, j);
        s += a[j].x * smlTaX[o] + a[j].y * smlTaX[o + 1u] + a[j].z * smlTaX[o + 2u] + a[j].w * smlTaX[o + 3u];
    }
    return s;
}

[ForceInline]
float sml_ta_dot_y(float4 a[4], uint r, uint lane4)
{
    float s = 0.0f;
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
    {
        uint o = r * HD + sml_ta_dim(lane4, j);
        s += a[j].x * smlTaY[o] + a[j].y * smlTaY[o + 1u] + a[j].z * smlTaY[o + 2u] + a[j].w * smlTaY[o + 3u];
    }
    return s;
}

[ForceInline]
void sml_ta_axpy_x(inout float4 acc[4], float w, uint r, uint lane4)
{
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
    {
        uint o = r * HD + sml_ta_dim(lane4, j);
        acc[j] += w * float4(smlTaX[o], smlTaX[o + 1u], smlTaX[o + 2u], smlTaX[o + 3u]);
    }
}

[ForceInline]
void sml_ta_axpy_y(inout float4 acc[4], float w, uint r, uint lane4)
{
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
    {
        uint o = r * HD + sml_ta_dim(lane4, j);
        acc[j] += w * float4(smlTaY[o], smlTaY[o + 1u], smlTaY[o + 2u], smlTaY[o + 3u]);
    }
}

// Stage 32 rows (positions r0..r0+31 of batch b) of two 64-wide head slices
// into smlTaX / smlTaY. Each of the 256 threads copies 2 float4 per slice.
[ForceInline]
void sml_ta_stage(float* base, uint ld, uint colX, uint colY, uint r0, uint tid)
{
    [ForceUnroll]
    for (uint i = 0; i < 2u; ++i)
    {
        uint idx = tid + 256u * i;  // 0..511
        uint r = idx >> 4u;
        uint q = idx & 15u;
        float* row = base + (r0 + r) * ld;
        float4 x = sml_ld_f4(reinterpret<float4*>(row + colX + 4u * q));
        float4 y = sml_ld_f4(reinterpret<float4*>(row + colY + 4u * q));
        uint o = r * HD + 4u * q;
        smlTaX[o] = x.x; smlTaX[o + 1u] = x.y; smlTaX[o + 2u] = x.z; smlTaX[o + 3u] = x.w;
        smlTaY[o] = y.x; smlTaY[o + 1u] = y.y; smlTaY[o + 2u] = y.z; smlTaY[o + 3u] = y.w;
    }
}

[ForceInline]
bool sml_ta_valid(uint i, uint j, uint window)
{
    return j <= i && (window == 0u || i - j < window);
}

// ---------------------------------------------------------------------------
// Forward. Grid: x = T / 64, y = Hq, z = batch.
// Tensor table: 0 qkv, 1 sinks [Hq], 2 O [rows][ldO], 3 L [rows][Hq].
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_attn_fwd_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTAttnPush.headerAddress);
    uint tid = tid3.x;
    uint row = tid >> 2u;
    uint lane4 = tid & 3u;
    uint h = gid.y;
    uint b = gid.z;
    uint kvh = h / (p.Hq / p.Hkv);
    uint i0 = gid.x * 64u;
    uint i = i0 + row;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    uint n = b * p.T + i;

    float4 q[4];
    sml_ta_load16(qkv + i * p.ldQKV + h * HD, lane4, q);
    float m = sml_tf(p.header, 1u)[h];
    float l = 1.0f;
    float4 o[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

    uint kStart = (p.window != 0u && i0 + 1u > p.window) ? ((i0 + 1u - p.window) & ~(TILE - 1u)) : 0u;
    uint kEnd = i0 + 64u;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;
    for (uint kt = kStart; kt < kEnd; kt += TILE)
    {
        sml_ta_stage(qkv, p.ldQKV, colK, colV, kt, tid);
        GroupMemoryBarrierWithGroupSync();
        for (uint jj = 0; jj < TILE; ++jj)
        {
            float s = sml_cluster_sum(sml_ta_dot_x(q, jj, lane4), 4u) * p.scale;
            if (sml_ta_valid(i, kt + jj, p.window))
            {
                if (s > m)
                {
                    float a = exp(m - s);
                    l = l * a + 1.0f;
                    [ForceUnroll]
                    for (uint j = 0; j < 4u; ++j)
                        o[j] *= a;
                    sml_ta_axpy_y(o, 1.0f, jj, lane4);
                    m = s;
                }
                else
                {
                    float e = exp(s - m);
                    l += e;
                    sml_ta_axpy_y(o, e, jj, lane4);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float inv = 1.0f / l;
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
        o[j] *= inv;
    sml_ta_store16(sml_tf(p.header, 2u) + n * p.ldO + h * HD, lane4, o);
    if (lane4 == 0u)
        sml_tf(p.header, 3u)[n * p.Hq + h] = m + log(l);
}

// ---------------------------------------------------------------------------
// Backward, query side: dq, D and the per-row sink-gradient terms.
// Grid: x = T / 64, y = Hq, z = batch.
// Tensor table: 0 qkv, 1 sinks, 2 O, 3 dO [rows][ldO], 4 L, 5 dqkv (q part
// written), 6 D [rows][Hq], 7 sinkTerm [rows][Hq] (= -exp(sink - L) D).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_attn_bwd_dq_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTAttnPush.headerAddress);
    uint tid = tid3.x;
    uint row = tid >> 2u;
    uint lane4 = tid & 3u;
    uint h = gid.y;
    uint b = gid.z;
    uint kvh = h / (p.Hq / p.Hkv);
    uint i0 = gid.x * 64u;
    uint i = i0 + row;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    uint n = b * p.T + i;

    float4 q[4], dO[4], ov[4];
    sml_ta_load16(qkv + i * p.ldQKV + h * HD, lane4, q);
    sml_ta_load16(sml_tf(p.header, 3u) + n * p.ldO + h * HD, lane4, dO);
    sml_ta_load16(sml_tf(p.header, 2u) + n * p.ldO + h * HD, lane4, ov);
    float dpart = 0.0f;
    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
        dpart += dot(dO[j], ov[j]);
    float Dv = sml_cluster_sum(dpart, 4u);
    float L = sml_tf(p.header, 4u)[n * p.Hq + h];
    float4 dq[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

    uint kStart = (p.window != 0u && i0 + 1u > p.window) ? ((i0 + 1u - p.window) & ~(TILE - 1u)) : 0u;
    uint kEnd = i0 + 64u;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;
    for (uint kt = kStart; kt < kEnd; kt += TILE)
    {
        sml_ta_stage(qkv, p.ldQKV, colK, colV, kt, tid);
        GroupMemoryBarrierWithGroupSync();
        for (uint jj = 0; jj < TILE; ++jj)
        {
            float s = sml_cluster_sum(sml_ta_dot_x(q, jj, lane4), 4u) * p.scale;
            float dp = sml_cluster_sum(sml_ta_dot_y(dO, jj, lane4), 4u);
            if (sml_ta_valid(i, kt + jj, p.window))
            {
                float pr = exp(s - L);
                sml_ta_axpy_x(dq, pr * (dp - Dv), jj, lane4);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll]
    for (uint j = 0; j < 4u; ++j)
        dq[j] *= p.scale;
    sml_ta_store16(sml_tf(p.header, 5u) + n * p.ldQKV + h * HD, lane4, dq);
    if (lane4 == 0u)
    {
        sml_tf(p.header, 6u)[n * p.Hq + h] = Dv;
        float sink = sml_tf(p.header, 1u)[h];
        sml_tf(p.header, 7u)[n * p.Hq + h] = -exp(sink - L) * Dv;
    }
}

// ---------------------------------------------------------------------------
// Backward, key side: dk and dv for one kv head, summed over its query heads.
// Grid: x = T / 64 (key blocks), y = Hkv, z = batch.
// Tensor table: 0 qkv, 1 dO, 2 L, 3 D, 4 dqkv (k and v parts written).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_attn_bwd_dkv_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainAttnParams* p = reinterpret<TrainAttnParams*>(smlTAttnPush.headerAddress);
    uint tid = tid3.x;
    uint row = tid >> 2u;
    uint lane4 = tid & 3u;
    uint kvh = gid.y;
    uint b = gid.z;
    uint G = p.Hq / p.Hkv;
    uint j0 = gid.x * 64u;
    uint j = j0 + row;
    float* qkv = sml_tf(p.header, 0u) + b * p.T * p.ldQKV;
    float* dOb = sml_tf(p.header, 1u) + b * p.T * p.ldO;
    float* Lb = sml_tf(p.header, 2u) + b * p.T * p.Hq;
    float* Db = sml_tf(p.header, 3u) + b * p.T * p.Hq;
    uint colK = (p.Hq + kvh) * HD;
    uint colV = (p.Hq + p.Hkv + kvh) * HD;

    float4 k[4], v[4];
    sml_ta_load16(qkv + j * p.ldQKV + colK, lane4, k);
    sml_ta_load16(qkv + j * p.ldQKV + colV, lane4, v);
    float4 dk[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
    float4 dv[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };

    uint qEnd = p.T;
    if (p.window != 0u)
        qEnd = min(p.T, (j0 + 64u + p.window - 1u + TILE - 1u) & ~(TILE - 1u));

    for (uint g = 0; g < G; ++g)
    {
        uint h = kvh * G + g;
        // Stage q (X) and dO (Y): q lives in qkv, dO in its own buffer, so stage them separately.
        for (uint qt = j0; qt < qEnd; qt += TILE)
        {
            [ForceUnroll]
            for (uint s2 = 0; s2 < 2u; ++s2)
            {
                uint idx = tid + 256u * s2;
                uint r = idx >> 4u;
                uint c4 = idx & 15u;
                float4 x = sml_ld_f4(reinterpret<float4*>(qkv + (qt + r) * p.ldQKV + h * HD + 4u * c4));
                float4 y = sml_ld_f4(reinterpret<float4*>(dOb + (qt + r) * p.ldO + h * HD + 4u * c4));
                uint o = r * HD + 4u * c4;
                smlTaX[o] = x.x; smlTaX[o + 1u] = x.y; smlTaX[o + 2u] = x.z; smlTaX[o + 3u] = x.w;
                smlTaY[o] = y.x; smlTaY[o + 1u] = y.y; smlTaY[o + 2u] = y.z; smlTaY[o + 3u] = y.w;
            }
            if (tid < TILE)
            {
                smlTaL[tid] = Lb[(qt + tid) * p.Hq + h];
                smlTaD[tid] = Db[(qt + tid) * p.Hq + h];
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint ii = 0; ii < TILE; ++ii)
            {
                float s = sml_cluster_sum(sml_ta_dot_x(k, ii, lane4), 4u) * p.scale;
                float dp = sml_cluster_sum(sml_ta_dot_y(v, ii, lane4), 4u);
                if (sml_ta_valid(qt + ii, j, p.window))
                {
                    float pr = exp(s - smlTaL[ii]);
                    sml_ta_axpy_y(dv, pr, ii, lane4);
                    sml_ta_axpy_x(dk, pr * (dp - smlTaD[ii]), ii, lane4);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    [ForceUnroll]
    for (uint c = 0; c < 4u; ++c)
        dk[c] *= p.scale;
    float* dqkv = sml_tf(p.header, 4u) + (b * p.T + j) * p.ldQKV;
    sml_ta_store16(dqkv + colK, lane4, dk);
    sml_ta_store16(dqkv + colV, lane4, dv);
}

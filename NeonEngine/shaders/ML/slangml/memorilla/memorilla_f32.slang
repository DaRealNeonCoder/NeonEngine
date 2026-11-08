// Memorilla memory-module kernels (docs/memorilla_plan.md). All fp32.
//
// The linear layers of the module use the training GEMMs (train/gemm_train_*.slang)
// and the bias / norm gradients use sml_train_colsum_f32; this file has the rest:
//   LayerNorm forward / backward           one 256-thread workgroup per row
//   exact GELU forward / backward          one thread per element
//   slot broadcast add + grouped row sums  (query conditioning, Stage 2)
//   cross-attention pieces for K slots over N documents, per head:
//     rowdot   W[b,h,i,j] = scale * <A[b,i,h,:], B[b,j,h,:]>     (-inf where j >= count[b])
//     softmax  rows of W in place; backward dS = scale * P * (dP - sum P dP)
//     wsum     Y[b,i,h,:] (+)= sum_j W[b,h,i,j] X[b,j,h,:]      (or W[b,h,j,i] when transposed)
//   strided row copy (memory rows into / out of the decoder's input rows).
// Tensors are row-major; head h of a row occupies columns [h*dh, (h+1)*dh).
import slangml_common;
import slangml_llm;
import slangml_train;

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlMemPush;

static const uint SML_MEM_ACCUM = 1u;
static const uint SML_MEM_MASK = 2u;
static const uint SML_MEM_TRANSPOSED = 4u;
static const uint SML_MEM_NO_DX = 8u;

groupshared float smlMemRed[32];

float sml_mem_sum(float v, uint tid)
{
    float s = WaveActiveSum(v);
    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
        smlMemRed[tid / lanes] = s;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    for (uint i = 0; i < 256u / lanes; ++i)
        total += smlMemRed[i];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

float sml_mem_max(float v, uint tid)
{
    float s = WaveActiveMax(v);
    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
        smlMemRed[tid / lanes] = s;
    GroupMemoryBarrierWithGroupSync();
    float total = -3.0e38f;
    for (uint i = 0; i < 256u / lanes; ++i)
        total = max(total, smlMemRed[i]);
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// ---------------------------------------------------------------------------
// LayerNorm. Forward: y = (x - mean) * rstd * gamma + beta, saves mean / rstd.
// Tensor table: 0 x [rows][H], 1 gamma, 2 beta, 3 y, 4 mean [rows], 5 rstd [rows].
// Backward: xhat = (x - mean) * rstd, g = gamma * dy,
//   dx (+)= rstd * (g - mean(g) - xhat * mean(g * xhat))     (skipped with NO_DX)
//   gx = dy * xhat (per element, for the gamma gradient; beta's is colsum(dy))
// Tensor table: 0 x, 1 gamma, 2 mean, 3 rstd, 4 dy, 5 dx, 6 gx.  Grid x = rows.
// ---------------------------------------------------------------------------
struct MemNormParams
{
    KernelHeader header;
    uint rows;   // 48
    uint H;      // 52
    float eps;   // 56
    uint flags;  // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_layernorm_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MemNormParams* p = reinterpret<MemNormParams*>(smlMemPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* x = sml_tf(p.header, 0u) + row * p.H;
    float* g = sml_tf(p.header, 1u);
    float* b = sml_tf(p.header, 2u);
    float* y = sml_tf(p.header, 3u) + row * p.H;
    float s = 0.0f;
    for (uint c = tid.x; c < p.H; c += 256u)
        s += x[c];
    float mean = sml_mem_sum(s, tid.x) / float(p.H);
    float v = 0.0f;
    for (uint c = tid.x; c < p.H; c += 256u)
    {
        float d = x[c] - mean;
        v += d * d;
    }
    float rstd = rsqrt(sml_mem_sum(v, tid.x) / float(p.H) + p.eps);
    for (uint c = tid.x; c < p.H; c += 256u)
        y[c] = (x[c] - mean) * rstd * g[c] + b[c];
    if (tid.x == 0u)
    {
        sml_tf(p.header, 4u)[row] = mean;
        sml_tf(p.header, 5u)[row] = rstd;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_layernorm_bwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MemNormParams* p = reinterpret<MemNormParams*>(smlMemPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* x = sml_tf(p.header, 0u) + row * p.H;
    float* gamma = sml_tf(p.header, 1u);
    float mean = sml_tf(p.header, 2u)[row];
    float rstd = sml_tf(p.header, 3u)[row];
    float* dy = sml_tf(p.header, 4u) + row * p.H;
    float* gx = sml_tf(p.header, 6u) + row * p.H;
    float s1 = 0.0f, s2 = 0.0f;
    for (uint c = tid.x; c < p.H; c += 256u)
    {
        float xh = (x[c] - mean) * rstd;
        float g = gamma[c] * dy[c];
        s1 += g;
        s2 += g * xh;
        gx[c] = dy[c] * xh;
    }
    if ((p.flags & SML_MEM_NO_DX) != 0u)
        return;  // uniform per workgroup: no barrier is skipped by only some threads
    float m1 = sml_mem_sum(s1, tid.x) / float(p.H);
    float m2 = sml_mem_sum(s2, tid.x) / float(p.H);
    float* dx = sml_tf(p.header, 5u) + row * p.H;
    for (uint c = tid.x; c < p.H; c += 256u)
    {
        float xh = (x[c] - mean) * rstd;
        float d = rstd * (gamma[c] * dy[c] - m1 - xh * m2);
        dx[c] = (p.flags & SML_MEM_ACCUM) != 0u ? dx[c] + d : d;
    }
}

// ---------------------------------------------------------------------------
// Exact GELU, y = 0.5 x (1 + erf(x / sqrt 2)). Forward: 0 x, 1 y.
// Backward: 0 x (the pre-activation), 1 dy -> dx in place.
// Grid x = ceil(n / 256) capped; threads stride by `stride`.
// ---------------------------------------------------------------------------
struct MemElemParams
{
    KernelHeader header;
    uint n;       // 48
    uint stride;  // 52
    uint r0;      // 56
    uint r1;      // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_gelu_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemElemParams* p = reinterpret<MemElemParams*>(smlMemPush.headerAddress);
    float* x = sml_tf(p.header, 0u);
    float* y = sml_tf(p.header, 1u);
    for (uint i = p.header.dispatchBase.x * 256u + dtid.x; i < p.n; i += p.stride)
        y[i] = slangml_gelu_exact(x[i]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_gelu_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemElemParams* p = reinterpret<MemElemParams*>(smlMemPush.headerAddress);
    float* x = sml_tf(p.header, 0u);
    float* d = sml_tf(p.header, 1u);
    for (uint i = p.header.dispatchBase.x * 256u + dtid.x; i < p.n; i += p.stride)
    {
        float v = x[i];
        float cdf = 0.5f * (1.0f + slangml_erf(v * SLANGML_INV_SQRT_2));
        float pdf = 0.3989422804014327f * exp(-0.5f * v * v);
        d[i] = d[i] * (cdf + v * pdf);
    }
}

// ---------------------------------------------------------------------------
// Slot broadcast add: out[b][k][c] = slots[k][c] + u[b][c].
// Tensor table: 0 slots [K][H], 1 u [B][H], 2 out [B][K][H]. Grid-stride over B*K*H.
// ---------------------------------------------------------------------------
struct MemBcastParams
{
    KernelHeader header;
    uint B;       // 48
    uint K;       // 52
    uint H;       // 56
    uint stride;  // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_bcast_add_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemBcastParams* p = reinterpret<MemBcastParams*>(smlMemPush.headerAddress);
    float* slots = sml_tf(p.header, 0u);
    float* u = sml_tf(p.header, 1u);
    float* out = sml_tf(p.header, 2u);
    uint n = p.B * p.K * p.H;
    for (uint i = p.header.dispatchBase.x * 256u + dtid.x; i < n; i += p.stride)
    {
        uint c = i % p.H;
        uint k = (i / p.H) % p.K;
        uint b = i / (p.H * p.K);
        out[i] = slots[k * p.H + c] + u[b * p.H + c];
    }
}

// ---------------------------------------------------------------------------
// Grouped row sums: out[g][c] (+)= sum_{r < R} in[g * groupStride + r * rowStride + c].
// Tensor table: 0 in, 1 out [G][C]. Grid (ceil(C / 256), G).
// ---------------------------------------------------------------------------
struct MemGroupSumParams
{
    KernelHeader header;
    uint G;            // 48
    uint R;            // 52
    uint C;            // 56
    uint groupStride;  // 60
    uint rowStride;    // 64
    uint flags;        // 68
    uint r0;           // 72
    uint r1;           // 76
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_group_sum_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MemGroupSumParams* p = reinterpret<MemGroupSumParams*>(smlMemPush.headerAddress);
    uint c = (p.header.dispatchBase.x + gid.x) * 256u + tid.x;
    uint g = p.header.dispatchBase.y + gid.y;
    if (c >= p.C || g >= p.G)
        return;
    float* in = sml_tf(p.header, 0u) + g * p.groupStride + c;
    float s = 0.0f;
    for (uint r = 0; r < p.R; ++r)
        s += in[r * p.rowStride];
    float* out = sml_tf(p.header, 1u) + g * p.C + c;
    out[0] = (p.flags & SML_MEM_ACCUM) != 0u ? out[0] + s : s;
}

// ---------------------------------------------------------------------------
// Per-head row dot products: W[b][h][i][j] = scale * sum_d A[(b*I + i)*ldA + h*dh + d]
//                                                        * B[(b*J + j)*ldB + h*dh + d]
// With MASK, entries with j >= count[b] are -inf (padded documents).
// Tensor table: 0 A, 1 B, 2 W [B][nh][I][J], 3 count (uint) [B]. Grid-stride over B*nh*I*J.
// ---------------------------------------------------------------------------
struct MemAttnParams
{
    KernelHeader header;
    uint B;       // 48
    uint I;       // 52
    uint J;       // 56
    uint nh;      // 60
    uint dh;      // 64
    uint ldA;     // 68  row stride of A (rowdot) / of Y (wsum)
    uint ldB;     // 72  row stride of B (rowdot) / of X (wsum)
    float scale;  // 76
    uint flags;   // 80
    uint stride;  // 84
    uint r0;      // 88
    uint r1;      // 92
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_rowdot_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemAttnParams* p = reinterpret<MemAttnParams*>(smlMemPush.headerAddress);
    float* A = sml_tf(p.header, 0u);
    float* Bm = sml_tf(p.header, 1u);
    float* W = sml_tf(p.header, 2u);
    uint n = p.B * p.nh * p.I * p.J;
    for (uint t = p.header.dispatchBase.x * 256u + dtid.x; t < n; t += p.stride)
    {
        uint j = t % p.J;
        uint i = (t / p.J) % p.I;
        uint h = (t / (p.J * p.I)) % p.nh;
        uint b = t / (p.J * p.I * p.nh);
        if ((p.flags & SML_MEM_MASK) != 0u && j >= sml_tu(p.header, 3u)[b])
        {
            W[t] = -3.0e38f;
            continue;
        }
        float* a = A + (b * p.I + i) * p.ldA + h * p.dh;
        float* bb = Bm + (b * p.J + j) * p.ldB + h * p.dh;
        float s = 0.0f;
        for (uint d = 0; d < p.dh; ++d)
            s += a[d] * bb[d];
        W[t] = s * p.scale;
    }
}

// ---------------------------------------------------------------------------
// Row softmax over J in place (rows = B*nh*I). Entries <= -1e38 count as masked
// and get probability 0; a fully masked row becomes all zeros.
// Tensor table: 0 W [rows][J]. Grid x = rows.
// Backward: dS = scale * P * (dP - sum_j P dP), written over dP.
// Tensor table: 0 P, 1 dP (in) / dS (out).
// ---------------------------------------------------------------------------
struct MemSoftmaxParams
{
    KernelHeader header;
    uint rows;    // 48
    uint J;       // 52
    float scale;  // 56
    uint r0;      // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_softmax_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MemSoftmaxParams* p = reinterpret<MemSoftmaxParams*>(smlMemPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* w = sml_tf(p.header, 0u) + row * p.J;
    float m = -3.0e38f;
    for (uint j = tid.x; j < p.J; j += 256u)
        m = max(m, w[j]);
    m = sml_mem_max(m, tid.x);
    float s = 0.0f;
    for (uint j = tid.x; j < p.J; j += 256u)
        s += w[j] > -1.0e38f ? exp(w[j] - m) : 0.0f;
    s = sml_mem_sum(s, tid.x);
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (uint j = tid.x; j < p.J; j += 256u)
        w[j] = w[j] > -1.0e38f ? exp(w[j] - m) * inv : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_softmax_bwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MemSoftmaxParams* p = reinterpret<MemSoftmaxParams*>(smlMemPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* P = sml_tf(p.header, 0u) + row * p.J;
    float* d = sml_tf(p.header, 1u) + row * p.J;
    float s = 0.0f;
    for (uint j = tid.x; j < p.J; j += 256u)
        s += P[j] * d[j];
    s = sml_mem_sum(s, tid.x);
    for (uint j = tid.x; j < p.J; j += 256u)
        d[j] = p.scale * P[j] * (d[j] - s);
}

// ---------------------------------------------------------------------------
// Weighted sums: Y[(b*I + i)*ldY + h*dh + d] (+)= sum_{j < J} w * X[(b*J + j)*ldX + h*dh + d]
// with w = W[b][h][i][j] (W stored [B][nh][I][J]) or, with TRANSPOSED,
// w = W[b][h][j][i] (W stored [B][nh][J][I]).
// Tensor table: 0 W, 1 X, 2 Y. ldA = ldY, ldB = ldX. Grid-stride over B*I*nh*dh.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_wsum_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemAttnParams* p = reinterpret<MemAttnParams*>(smlMemPush.headerAddress);
    float* W = sml_tf(p.header, 0u);
    float* X = sml_tf(p.header, 1u);
    float* Y = sml_tf(p.header, 2u);
    uint width = p.nh * p.dh;
    uint n = p.B * p.I * width;
    bool tr = (p.flags & SML_MEM_TRANSPOSED) != 0u;
    for (uint t = p.header.dispatchBase.x * 256u + dtid.x; t < n; t += p.stride)
    {
        uint c = t % width;
        uint i = (t / width) % p.I;
        uint b = t / (width * p.I);
        uint h = c / p.dh;
        float* w = W + (b * p.nh + h) * p.I * p.J;
        float* x = X + b * p.J * p.ldB + c;
        float s = 0.0f;
        if (tr)
        {
            for (uint j = 0; j < p.J; ++j)
                s += w[j * p.I + i] * x[j * p.ldB];
        }
        else
        {
            float* wr = w + i * p.J;
            for (uint j = 0; j < p.J; ++j)
                s += wr[j] * x[j * p.ldB];
        }
        float* y = Y + (b * p.I + i) * p.ldA + c;
        y[0] = (p.flags & SML_MEM_ACCUM) != 0u ? y[0] + s : s;
    }
}

// ---------------------------------------------------------------------------
// Strided row copy: out[(b*outT + outOff + k)*H + c] (+)= in[(b*inT + inOff + k)*H + c]
// for b < B, k < K. Used to place memory rows in the decoder input and to read
// their gradient back. Tensor table: 0 in, 1 out. Grid-stride over B*K*H.
// ---------------------------------------------------------------------------
struct MemRowsParams
{
    KernelHeader header;
    uint B;       // 48
    uint K;       // 52
    uint H;       // 56
    uint inT;     // 60
    uint inOff;   // 64
    uint outT;    // 68
    uint outOff;  // 72
    uint flags;   // 76
    uint stride;  // 80
    uint r0;      // 84
    uint r1;      // 88
    uint r2;      // 92
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_mem_rows_copy_f32(uint3 dtid : SV_DispatchThreadID)
{
    MemRowsParams* p = reinterpret<MemRowsParams*>(smlMemPush.headerAddress);
    float* in = sml_tf(p.header, 0u);
    float* out = sml_tf(p.header, 1u);
    uint n = p.B * p.K * p.H;
    for (uint t = p.header.dispatchBase.x * 256u + dtid.x; t < n; t += p.stride)
    {
        uint c = t % p.H;
        uint k = (t / p.H) % p.K;
        uint b = t / (p.H * p.K);
        float v = in[(b * p.inT + p.inOff + k) * p.H + c];
        uint o = (b * p.outT + p.outOff + k) * p.H + c;
        out[o] = (p.flags & SML_MEM_ACCUM) != 0u ? out[o] + v : v;
    }
}

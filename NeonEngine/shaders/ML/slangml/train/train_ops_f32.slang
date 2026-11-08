// Training ops for the gpt-oss trainer (docs/gpt_oss_training.md): embedding,
// RMSNorm, RoPE, column sums, cross-entropy, gradient norm, AdamW and MXFP4
// fake quantization. All fp32. Row kernels use one 256-thread workgroup per
// row; elementwise kernels one thread per element (grid x = ceil(n / 256)).
import slangml_common;
import slangml_llm;
import slangml_train;

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTrainPush;

groupshared float smlTrRed[32];

// Sum / max over the 256-thread workgroup; every thread gets the result.
float sml_tr_sum(float v, uint tid)
{
    float s = WaveActiveSum(v);
    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
        smlTrRed[tid / lanes] = s;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    for (uint i = 0; i < 256u / lanes; ++i)
        total += smlTrRed[i];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

float sml_tr_max(float v, uint tid)
{
    float s = WaveActiveMax(v);
    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
        smlTrRed[tid / lanes] = s;
    GroupMemoryBarrierWithGroupSync();
    float total = -3.0e38f;
    for (uint i = 0; i < 256u / lanes; ++i)
        total = max(total, smlTrRed[i]);
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// ---------------------------------------------------------------------------
// Embedding. Forward: x[n] = E[tok[n]]. Grid x = rows.
// Tensor table: 0 tokens (uint) [rows], 1 E [V][H], 2 x [rows][H].
// Backward (one workgroup per distinct token u, CSR built on the host):
//   dE[tok[u]] += sum_{j in [off[u], off[u+1])} dx[pos[j]]
// Tensor table: 0 uniqueTok [U], 1 off [U+1], 2 pos, 3 dx [rows][H], 4 dE.
// ---------------------------------------------------------------------------
struct EmbedParams
{
    KernelHeader header;
    uint rows;  // 48 (rows, or U for the backward)
    uint H;     // 52
    uint r0;    // 56
    uint r1;    // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_embed_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    EmbedParams* p = reinterpret<EmbedParams*>(smlTrainPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    uint tok = sml_tu(p.header, 0u)[row];
    float* E = sml_tf(p.header, 1u) + tok * p.H;
    float* x = sml_tf(p.header, 2u) + row * p.H;
    for (uint c = tid.x; c < p.H; c += 256u)
        x[c] = E[c];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_embed_bwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    EmbedParams* p = reinterpret<EmbedParams*>(smlTrainPush.headerAddress);
    uint u = p.header.dispatchBase.x + gid.x;
    if (u >= p.rows)
        return;
    uint tok = sml_tu(p.header, 0u)[u];
    uint* off = sml_tu(p.header, 1u);
    uint* pos = sml_tu(p.header, 2u);
    float* dx = sml_tf(p.header, 3u);
    float* dE = sml_tf(p.header, 4u) + tok * p.H;
    uint b = off[u];
    uint e = off[u + 1u];
    for (uint c = tid.x; c < p.H; c += 256u)
    {
        float s = 0.0f;
        for (uint j = b; j < e; ++j)
            s += dx[pos[j] * p.H + c];
        dE[c] += s;
    }
}

// ---------------------------------------------------------------------------
// RMSNorm (gpt-oss: y = x * rsqrt(mean(x^2) + eps) * w, fp32 scale).
// Forward. Grid x = rows. Tensor table: 0 x, 1 w, 2 y, 3 rstd [rows].
// Backward, with g = dy * w and r = rstd:
//   dx = r * g - x * r^3 * sum(g * x) / H     (added to dX, or written with OVERWRITE)
//   dy is overwritten in place with dy * x * r (the per-row dw terms; the
//   host column-sums them into dw).
// Tensor table: 0 x, 1 w, 2 rstd, 3 dy (in: grad of y, out: dw rows), 4 dX.
// ---------------------------------------------------------------------------
static const uint SML_TNORM_OVERWRITE = 1u;

struct TrainNormParams
{
    KernelHeader header;
    uint rows;    // 48
    uint H;       // 52
    float eps;    // 56
    uint flags;   // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_rmsnorm_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    TrainNormParams* p = reinterpret<TrainNormParams*>(smlTrainPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* x = sml_tf(p.header, 0u) + row * p.H;
    float* w = sml_tf(p.header, 1u);
    float* y = sml_tf(p.header, 2u) + row * p.H;
    float sq = 0.0f;
    for (uint c = tid.x; c < p.H; c += 256u)
        sq += x[c] * x[c];
    float r = rsqrt(sml_tr_sum(sq, tid.x) / float(p.H) + p.eps);
    for (uint c = tid.x; c < p.H; c += 256u)
        y[c] = x[c] * r * w[c];
    if (tid.x == 0u)
        sml_tf(p.header, 3u)[row] = r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_rmsnorm_bwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    TrainNormParams* p = reinterpret<TrainNormParams*>(smlTrainPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* x = sml_tf(p.header, 0u) + row * p.H;
    float* w = sml_tf(p.header, 1u);
    float r = sml_tf(p.header, 2u)[row];
    float* dy = sml_tf(p.header, 3u) + row * p.H;
    float* dX = sml_tf(p.header, 4u) + row * p.H;
    float s = 0.0f;
    for (uint c = tid.x; c < p.H; c += 256u)
        s += dy[c] * w[c] * x[c];
    s = sml_tr_sum(s, tid.x);
    float k = r * r * r * s / float(p.H);
    bool overwrite = (p.flags & SML_TNORM_OVERWRITE) != 0u;
    for (uint c = tid.x; c < p.H; c += 256u)
    {
        float g = dy[c];
        float xc = x[c];
        float d = r * g * w[c] - xc * k;
        dX[c] = overwrite ? d : dX[c] + d;
        dy[c] = g * xc * r;
    }
}

// ---------------------------------------------------------------------------
// RoPE on the q and k parts of a packed qkv buffer [rows][ld], in place.
// Row n is position n % T. NeoX halves: for i < D/2,
//   fwd: o1 = x1 c - x2 s, o2 = x2 c + x1 s
//   bwd (transpose): d1 = g1 c + g2 s, d2 = g2 c - g1 s
// c, s come from host tables [T][D/2] that already include the YaRN
// concentration. One thread per (row, head, i), grid-stride (grid x <= 65535).
// Tensor table: 0 qkv, 1 cos, 2 sin.
// ---------------------------------------------------------------------------
struct TrainRopeParams
{
    KernelHeader header;
    uint rows;     // 48
    uint T;        // 52
    uint heads;    // 56  Hq + Hkv (q heads then k heads, contiguous)
    uint D;        // 60
    uint ld;       // 64
    uint inverse;  // 68
    uint stride;   // 72  grid-stride in threads
    uint reserved0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_rope_f32(uint3 dtid : SV_DispatchThreadID)
{
    TrainRopeParams* p = reinterpret<TrainRopeParams*>(smlTrainPush.headerAddress);
    uint half = p.D >> 1u;
    uint perRow = p.heads * half;
    for (uint idx = dtid.x; idx < p.rows * perRow; idx += p.stride)
    {
    uint row = idx / perRow;
    uint rem = idx - row * perRow;
    uint head = rem / half;
    uint i = rem - head * half;
    uint pos = row % p.T;
    float c = sml_tf(p.header, 1u)[pos * half + i];
    float s = sml_tf(p.header, 2u)[pos * half + i];
    float* v = sml_tf(p.header, 0u) + row * p.ld + head * p.D;
    float x1 = v[i];
    float x2 = v[i + half];
    if (p.inverse != 0u)
    {
        v[i] = x1 * c + x2 * s;
        v[i + half] = x2 * c - x1 * s;
    }
    else
    {
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
    }
    }
}

// ---------------------------------------------------------------------------
// Column sums: out[g * strideOut + c] (+)= sum_{r in rows of g} in[r * ld + c].
// Rows of group g: [seg[g], seg[g+1]) when SEGMENTED, else [0, rows).
// 64 columns x 4 row lanes per workgroup. Grid: x = ceil(cols / 64), z = groups.
// Tensor table: 0 in, 1 out, 2 seg (uint, SEGMENTED only).
// ---------------------------------------------------------------------------
static const uint SML_TCOL_ACCUM = 1u;
static const uint SML_TCOL_SEGMENTED = 2u;

struct ColSumParams
{
    KernelHeader header;
    uint rows;       // 48
    uint cols;       // 52
    uint ld;         // 56
    uint flags;      // 60
    uint strideOut;  // 64
    uint reserved0;  // 68
};

groupshared float smlColPart[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_colsum_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    ColSumParams* p = reinterpret<ColSumParams*>(smlTrainPush.headerAddress);
    uint lc = tid.x & 63u;
    uint lr = tid.x >> 6u;
    uint c = gid.x * 64u + lc;
    uint g = gid.z;
    uint r0 = 0u;
    uint r1 = p.rows;
    if ((p.flags & SML_TCOL_SEGMENTED) != 0u)
    {
        uint* seg = sml_tu(p.header, 2u);
        r0 = seg[g];
        r1 = seg[g + 1u];
    }
    float* src = sml_tf(p.header, 0u);
    float s = 0.0f;
    if (c < p.cols)
    {
        for (uint r = r0 + lr; r < r1; r += 4u)
            s += src[r * p.ld + c];
    }
    smlColPart[tid.x] = s;
    GroupMemoryBarrierWithGroupSync();
    if (lr == 0u && c < p.cols)
    {
        float t = smlColPart[lc] + smlColPart[lc + 64u] + smlColPart[lc + 128u] + smlColPart[lc + 192u];
        float* out = sml_tf(p.header, 1u) + g * p.strideOut;
        out[c] = (p.flags & SML_TCOL_ACCUM) != 0u ? out[c] + t : t;
    }
}

// ---------------------------------------------------------------------------
// Fused softmax cross-entropy over a chunk of rows of logits [rows][V]:
//   loss[r] = logsumexp(z_r) - z_r[t_r]
//   z_r     <- (softmax(z_r) - onehot(t_r)) * scale       (the logit gradient)
// Rows with t_r == SML_INVALID get loss 0 and gradient 0. Grid x = rows.
// Tensor table: 0 logits (in/out), 1 targets (uint) [rows], 2 loss [rows].
// ---------------------------------------------------------------------------
struct CrossEntropyParams
{
    KernelHeader header;
    uint rows;    // 48
    uint V;       // 52
    float scale;  // 56
    uint reserved0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_xent_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    CrossEntropyParams* p = reinterpret<CrossEntropyParams*>(smlTrainPush.headerAddress);
    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    float* z = sml_tf(p.header, 0u) + row * p.V;
    uint target = sml_tu(p.header, 1u)[row];
    if (target == SML_INVALID)
    {
        for (uint c = tid.x; c < p.V; c += 256u)
            z[c] = 0.0f;
        if (tid.x == 0u)
            sml_tf(p.header, 2u)[row] = 0.0f;
        return;
    }
    // Online max / sum per thread, then combined.
    float m = -3.0e38f;
    float l = 0.0f;
    for (uint c = tid.x; c < p.V; c += 256u)
    {
        float v = z[c];
        if (v > m)
        {
            l = l * exp(m - v) + 1.0f;
            m = v;
        }
        else
        {
            l += exp(v - m);
        }
    }
    float gm = sml_tr_max(m, tid.x);
    float gl = sml_tr_sum(l * exp(m - gm), tid.x);
    float lse = gm + log(gl);
    float zt = z[target];
    float inv = 1.0f / gl;
    for (uint c = tid.x; c < p.V; c += 256u)
    {
        float pr = exp(z[c] - gm) * inv;
        z[c] = (pr - (c == target ? 1.0f : 0.0f)) * p.scale;
    }
    if (tid.x == 0u)
        sml_tf(p.header, 2u)[row] = lse - zt;
}

// ---------------------------------------------------------------------------
// Sum of squares: pass 1 writes one partial per workgroup (each covers
// 256 * itemsPerThread elements); pass 2 (one workgroup) sums `n` partials
// into out[0] (+)= total. Tensor table: 0 in, 1 out.
// ---------------------------------------------------------------------------
struct SumSqParams
{
    KernelHeader header;
    uint n;               // 48
    uint itemsPerThread;  // 52
    uint square;          // 56  1 = sum of squares, 0 = plain sum
    uint accum;           // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_sumsq_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SumSqParams* p = reinterpret<SumSqParams*>(smlTrainPush.headerAddress);
    float* src = sml_tf(p.header, 0u);
    uint base = (p.header.dispatchBase.x + gid.x) * 256u * p.itemsPerThread;
    float s = 0.0f;
    for (uint i = 0; i < p.itemsPerThread; ++i)
    {
        uint idx = base + i * 256u + tid.x;
        if (idx < p.n)
        {
            float v = src[idx];
            s += p.square != 0u ? v * v : v;
        }
    }
    s = sml_tr_sum(s, tid.x);
    if (tid.x == 0u)
    {
        float* out = sml_tf(p.header, 1u) + gid.x;
        out[0] = p.accum != 0u ? out[0] + s : s;
    }
}

// ---------------------------------------------------------------------------
// AdamW with global-norm gradient clipping, fused with zeroing the gradient.
//   g = grad * gradScale * min(1, clip / (||grad * gradScale|| + 1e-6))   (||grad||^2 in normSq[0];
//   gradScale = 1 / (micro-batches * loss scale)). A non-finite normSq skips the update.
//   m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
//   p -= lr * (m / bc1) / (sqrt(v / bc2) + eps) + lr * wd * p   (wd only for i < decayEnd)
// Grid-stride (grid x <= 65535). Tensor table: 0 params, 1 grads, 2 m, 3 v, 4 normSq.
// ---------------------------------------------------------------------------
struct AdamParams
{
    KernelHeader header;
    uint n;          // 48
    uint decayEnd;   // 52
    float lr;        // 56
    float beta1;     // 60
    float beta2;     // 64
    float eps;       // 68
    float wd;        // 72
    float bc1;       // 76  1 - beta1^t
    float bc2;       // 80  1 - beta2^t
    float clip;      // 84  <= 0 disables clipping
    float gradScale; // 88  applied before clipping (e.g. 1 / accumulation steps)
    uint stride;     // 92  grid-stride in threads
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_adamw_f32(uint3 dtid : SV_DispatchThreadID)
{
    AdamParams* p = reinterpret<AdamParams*>(smlTrainPush.headerAddress);
    float* prm = sml_tf(p.header, 0u);
    float* grd = sml_tf(p.header, 1u);
    float* mm = sml_tf(p.header, 2u);
    float* vv = sml_tf(p.header, 3u);
    float scale = p.gradScale;
    float normSq = sml_tf(p.header, 4u)[0];
    if ((asuint(normSq) & 0x7F800000u) == 0x7F800000u)
    {
        // inf / NaN gradient (fp16 overflow under loss scaling): skip the update.
        for (uint i = dtid.x; i < p.n; i += p.stride)
            grd[i] = 0.0f;
        return;
    }
    if (p.clip > 0.0f)
    {
        float norm = sqrt(normSq) * p.gradScale;
        scale *= min(1.0f, p.clip / (norm + 1.0e-6f));
    }
    for (uint i = dtid.x; i < p.n; i += p.stride)
    {
    float g = grd[i] * scale;
    float m = p.beta1 * mm[i] + (1.0f - p.beta1) * g;
    float v = p.beta2 * vv[i] + (1.0f - p.beta2) * g * g;
    mm[i] = m;
    vv[i] = v;
    float w = prm[i];
    float upd = (m / p.bc1) / (sqrt(v / p.bc2) + p.eps);
    if (i < p.decayEnd)
        upd += p.wd * w;
    prm[i] = w - p.lr * upd;
    grd[i] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// MXFP4 fake quantization (quantization-aware training): out = dequant(quant(in))
// with the checkpoint rounding rule (slangml-train.slang). One thread per block
// of 32 consecutive elements, grid-stride (grid x <= 65535).
// Tensor table: 0 in, 1 out.
// ---------------------------------------------------------------------------
struct FakeQuantParams
{
    KernelHeader header;
    uint blocks;  // 48
    uint stride;  // 52 grid-stride in threads
    uint reserved1;
    uint reserved2;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_mxfp4_fakequant_f32(uint3 dtid : SV_DispatchThreadID)
{
    FakeQuantParams* p = reinterpret<FakeQuantParams*>(smlTrainPush.headerAddress);
    for (uint b = dtid.x; b < p.blocks; b += p.stride)
    {
    float4* src = sml_tf4(p.header, 0u) + b * 8u;
    float4* dst = sml_tf4(p.header, 1u) + b * 8u;
    float4 v[8];
    float amax = 0.0f;
    [ForceUnroll]
    for (uint i = 0; i < 8u; ++i)
    {
        v[i] = sml_ld_f4(src + i);
        amax = max(amax, max(max(abs(v[i].x), abs(v[i].y)), max(abs(v[i].z), abs(v[i].w))));
    }
    int e = sml_mxfp4_exponent(amax);
    float inv = sml_pow2(-e);
    float sc = sml_pow2(e);
    [ForceUnroll]
    for (uint i = 0; i < 8u; ++i)
    {
        float4 a = v[i] * inv;
        float4 q = float4(sml_e2m1_round(abs(a.x)), sml_e2m1_round(abs(a.y)),
                          sml_e2m1_round(abs(a.z)), sml_e2m1_round(abs(a.w)));
        q = float4(a.x < 0.0f ? -q.x : q.x, a.y < 0.0f ? -q.y : q.y,
                   a.z < 0.0f ? -q.z : q.z, a.w < 0.0f ? -q.w : q.w);
        sml_st_f4(dst + i, e <= -127 ? float4(0.0f) : q * sc);
    }
    }
}

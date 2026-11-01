import slangml_common;

// Backward pass for triangular gated self-attention [SI Alg 14, Alg 15].
// Forward is triangle_attention_f32.slang.  For an output row r = (i, j, h),
// with key index kk (see sml_tri_locate):
//
//   logit_{r,kk} = scale * q_r . K_{idx(r,kk)} + bias_{bidx(r,kk)}
//   A_{r,kk}     = softmax_kk(logit_r)
//   oAcc_r       = sum_kk A_{r,kk} V_{idx(r,kk)}
//   out_r        = G_r * oAcc_r,   G = sigmoid(gate)
//
// Gradients, with dOAcc_r = dOut_r * G_r and delta_r = dOAcc_r . oAcc_r:
//
//   dGate_r        = dOut_r * oAcc_r * G_r * (1 - G_r)
//   dlogit_{r,kk}  = A_{r,kk} * (dOAcc_r . V_{idx(r,kk)} - delta_r)
//   dQ_r           = scale * sum_kk dlogit_{r,kk} K_{idx(r,kk)}
//   dBias_b        = sum over rows r that read bias element b of dlogit_{r,kk}
//   dK_e           = scale * sum over rows r that read K element e of dlogit_{r,e} q_r
//   dV_e           = sum over rows r that read V element e of A_{r,e} dOAcc_r
//
// A bias, K, or V element is read by many output rows, so a direct scatter would
// need atomics.  The backward is therefore four gather passes.  Each thread owns
// one output element and loops over the single index it shares with others:
//
//   stats   thread per output row     : lse, oAcc, dOAcc, delta, dGate
//   dq      thread per output row     : dQ
//   dbias   thread per bias element   : dBias   (loops over the row index)
//   dkdv    thread per K/V element    : dK, dV  (loops over the row index)
//
// Each of the four runs for the starting node and for the ending node, so there are
// eight entry points.
//
// Layout (fp32, canonical):
//   q, k, v, gate, dOut, dQ, dK, dV, dGate, oAcc, dOAcc : [N, N, H, C]   index ((i*N + j)*H + h)*C + c
//   bias, dBias                                         : [N, N, H]      index (i*N + j)*H + h
//   lse, delta                                          : [N, N, H]      index (i*N + j)*H + h
//
// Workspace: lse, oAcc, dOAcc, delta are produced by the stats pass.
// Performance: the dbias and dkdv passes do N * N * H * N work.  Those are the
// hot spots and are the first candidates for tiling (see docs/af3_ops.md).

static const uint SML_TRIB_MAX_C = 32u;

struct TriangleAttentionBwdParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint C;
    float scale;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTriangleAttentionBwdPush;

float sml_trib_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_trib_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// For output row (ri, rj) and key kk, return the K/V base index and the bias index.
//   start (Alg 14): K/V at (ri, kk),  bias at (rj, kk)
//   end   (Alg 15): K/V at (kk, rj),  bias at (kk, ri)
void sml_trib_locate(bool ending, uint ri, uint rj, uint kk, uint N, uint H, uint C, uint h,
    out uint kvBase, out uint biasIndex)
{
    uint kvA = ending ? kk : ri;
    uint kvB = ending ? rj : kk;
    kvBase = ((kvA * N + kvB) * H + h) * C;
    biasIndex = ending ? ((kk * N + ri) * H + h) : ((rj * N + kk) * H + h);
}

// Stats: lse, oAcc, dOAcc, delta, dGate for one output row.
void sml_trib_stats_run(uint t, bool ending)
{
    TriangleAttentionBwdParams* p = reinterpret<TriangleAttentionBwdParams*>(smlTriangleAttentionBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc gate = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOut = slangml_tensor_desc(p.header, 5u);
    TensorDesc lse = slangml_tensor_desc(p.header, 6u);
    TensorDesc oAcc = slangml_tensor_desc(p.header, 7u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 8u);
    TensorDesc delta = slangml_tensor_desc(p.header, 9u);
    TensorDesc dGate = slangml_tensor_desc(p.header, 10u);

    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    if (t >= N * N * H)
        return;
    uint h = t % H;
    uint ij = t / H;
    uint rj = ij % N;
    uint ri = ij / N;

    float qv[SML_TRIB_MAX_C];
    float acc[SML_TRIB_MAX_C];
    uint rowBase = ((ri * N + rj) * H + h) * C;
    for (uint c = 0u; c < C; ++c)
    {
        qv[c] = sml_trib_load(q, rowBase + c);
        acc[c] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint kk = 0u; kk < N; ++kk)
    {
        uint kvBase;
        uint biasIndex;
        sml_trib_locate(ending, ri, rj, kk, N, H, C, h, kvBase, biasIndex);
        float dot = 0.0f;
        for (uint c = 0u; c < C; ++c)
            dot = fma(qv[c], sml_trib_load(k, kvBase + c), dot);
        float logit = fma(dot, p.scale, sml_trib_load(bias, biasIndex));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint c = 0u; c < C; ++c)
            acc[c] = fma(weight, sml_trib_load(v, kvBase + c), acc[c] * correction);
        runningMax = newMax;
    }

    uint rowIndex = (ri * N + rj) * H + h;
    float invSum = 1.0f / runningSum;
    sml_trib_store(lse, rowIndex, runningMax + log(runningSum));

    float deltaValue = 0.0f;
    for (uint c = 0u; c < C; ++c)
    {
        float g = slangml_sigmoid(sml_trib_load(gate, rowBase + c));
        float o = acc[c] * invSum;
        float dO = sml_trib_load(dOut, rowBase + c);
        float dOa = dO * g;
        sml_trib_store(oAcc, rowBase + c, o);
        sml_trib_store(dOAcc, rowBase + c, dOa);
        sml_trib_store(dGate, rowBase + c, dO * o * g * (1.0f - g));
        deltaValue = fma(dOa, o, deltaValue);
    }
    sml_trib_store(delta, rowIndex, deltaValue);
}

// dQ for one output row.
void sml_trib_dq_run(uint t, bool ending)
{
    TriangleAttentionBwdParams* p = reinterpret<TriangleAttentionBwdParams*>(smlTriangleAttentionBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dQ = slangml_tensor_desc(p.header, 7u);

    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    if (t >= N * N * H)
        return;
    uint h = t % H;
    uint ij = t / H;
    uint rj = ij % N;
    uint ri = ij / N;

    float qv[SML_TRIB_MAX_C];
    float dO[SML_TRIB_MAX_C];
    float dq[SML_TRIB_MAX_C];
    uint rowBase = ((ri * N + rj) * H + h) * C;
    uint rowIndex = (ri * N + rj) * H + h;
    for (uint c = 0u; c < C; ++c)
    {
        qv[c] = sml_trib_load(q, rowBase + c);
        dO[c] = sml_trib_load(dOAcc, rowBase + c);
        dq[c] = 0.0f;
    }
    float lseI = sml_trib_load(lse, rowIndex);
    float deltaI = sml_trib_load(delta, rowIndex);

    for (uint kk = 0u; kk < N; ++kk)
    {
        uint kvBase;
        uint biasIndex;
        sml_trib_locate(ending, ri, rj, kk, N, H, C, h, kvBase, biasIndex);
        float dot = 0.0f;
        float dA = 0.0f;
        for (uint c = 0u; c < C; ++c)
        {
            dot = fma(qv[c], sml_trib_load(k, kvBase + c), dot);
            dA = fma(dO[c], sml_trib_load(v, kvBase + c), dA);
        }
        float logit = fma(dot, p.scale, sml_trib_load(bias, biasIndex));
        float prob = exp(logit - lseI);
        float scaled = p.scale * prob * (dA - deltaI);
        for (uint c = 0u; c < C; ++c)
            dq[c] = fma(scaled, sml_trib_load(k, kvBase + c), dq[c]);
    }

    for (uint c = 0u; c < C; ++c)
        sml_trib_store(dQ, rowBase + c, dq[c]);
}

// dBias for one bias element (x, y, h).  The free index f runs over the row index
// that reads this element:
//   start: row (f, x), key y      end: row (y, f), key x
void sml_trib_dbias_run(uint t, bool ending)
{
    TriangleAttentionBwdParams* p = reinterpret<TriangleAttentionBwdParams*>(smlTriangleAttentionBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dBias = slangml_tensor_desc(p.header, 7u);

    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    if (t >= N * N * H)
        return;
    uint h = t % H;
    uint xy = t / H;
    uint x = xy / N;
    uint y = xy % N;
    uint biasElement = (x * N + y) * H + h;

    float acc = 0.0f;
    for (uint f = 0u; f < N; ++f)
    {
        uint ri = ending ? y : f;
        uint rj = ending ? f : x;
        uint kk = ending ? x : y;
        uint rowBase = ((ri * N + rj) * H + h) * C;
        uint rowIndex = (ri * N + rj) * H + h;
        uint kvBase;
        uint biasIndex;
        sml_trib_locate(ending, ri, rj, kk, N, H, C, h, kvBase, biasIndex);

        float dot = 0.0f;
        float dA = 0.0f;
        for (uint c = 0u; c < C; ++c)
        {
            dot = fma(sml_trib_load(q, rowBase + c), sml_trib_load(k, kvBase + c), dot);
            dA = fma(sml_trib_load(dOAcc, rowBase + c), sml_trib_load(v, kvBase + c), dA);
        }
        float logit = fma(dot, p.scale, sml_trib_load(bias, biasIndex));
        float prob = exp(logit - sml_trib_load(lse, rowIndex));
        acc += prob * (dA - sml_trib_load(delta, rowIndex));
    }
    sml_trib_store(dBias, biasElement, acc);
}

// dK and dV for one K/V element (x, y, h).  The free index f runs over the row index
// that reads this element:
//   start: row (x, f), key y      end: row (f, y), key x
void sml_trib_dkdv_run(uint t, bool ending)
{
    TriangleAttentionBwdParams* p = reinterpret<TriangleAttentionBwdParams*>(smlTriangleAttentionBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dK = slangml_tensor_desc(p.header, 7u);
    TensorDesc dV = slangml_tensor_desc(p.header, 8u);

    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    if (t >= N * N * H)
        return;
    uint h = t % H;
    uint xy = t / H;
    uint x = xy / N;
    uint y = xy % N;
    uint kvElement = ((x * N + y) * H + h) * C;

    float kv[SML_TRIB_MAX_C];
    float vv[SML_TRIB_MAX_C];
    float dk[SML_TRIB_MAX_C];
    float dv[SML_TRIB_MAX_C];
    for (uint c = 0u; c < C; ++c)
    {
        kv[c] = sml_trib_load(k, kvElement + c);
        vv[c] = sml_trib_load(v, kvElement + c);
        dk[c] = 0.0f;
        dv[c] = 0.0f;
    }

    for (uint f = 0u; f < N; ++f)
    {
        uint ri = ending ? f : x;
        uint rj = ending ? y : f;
        uint kk = ending ? x : y;
        uint rowBase = ((ri * N + rj) * H + h) * C;
        uint rowIndex = (ri * N + rj) * H + h;
        uint kvBase;
        uint biasIndex;
        sml_trib_locate(ending, ri, rj, kk, N, H, C, h, kvBase, biasIndex);

        float dot = 0.0f;
        float dA = 0.0f;
        for (uint c = 0u; c < C; ++c)
        {
            dot = fma(sml_trib_load(q, rowBase + c), kv[c], dot);
            dA = fma(sml_trib_load(dOAcc, rowBase + c), vv[c], dA);
        }
        float logit = fma(dot, p.scale, sml_trib_load(bias, biasIndex));
        float prob = exp(logit - sml_trib_load(lse, rowIndex));
        float scaled = p.scale * prob * (dA - sml_trib_load(delta, rowIndex));
        for (uint c = 0u; c < C; ++c)
        {
            dv[c] = fma(prob, sml_trib_load(dOAcc, rowBase + c), dv[c]);
            dk[c] = fma(scaled, sml_trib_load(q, rowBase + c), dk[c]);
        }
    }

    for (uint c = 0u; c < C; ++c)
    {
        sml_trib_store(dK, kvElement + c, dk[c]);
        sml_trib_store(dV, kvElement + c, dv[c]);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_start_bwd_f32_stats(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_stats_run(dtid.x, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_end_bwd_f32_stats(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_stats_run(dtid.x, true);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_start_bwd_f32_dq(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dq_run(dtid.x, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_end_bwd_f32_dq(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dq_run(dtid.x, true);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_start_bwd_f32_dbias(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dbias_run(dtid.x, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_end_bwd_f32_dbias(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dbias_run(dtid.x, true);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_start_bwd_f32_dkdv(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dkdv_run(dtid.x, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_end_bwd_f32_dkdv(uint3 dtid : SV_DispatchThreadID)
{
    sml_trib_dkdv_run(dtid.x, true);
}

import slangml_common;

// Backward pass for pair-biased single attention [SI Alg 24].  Forward is
// attention_pair_bias_fwd_f32:
//
//   oAcc[i,h,:] = sum_j A_ij v_j,   A_ij = softmax_j(scale * q_i.k_j + bias_ijh)
//   out[i,h,:]  = G[i,h,:] * oAcc[i,h,:],   G = sigmoid(gate)
//
// Gradients, with dO = dOut and dOAcc = dO * G:
//
//   dGate_i    = dO * oAcc * G * (1 - G)
//   delta_i    = dOAcc_i . oAcc_i            (= sum_j A_ij dA_ij)
//   dA_ij      = dOAcc_i . v_j
//   dlogit_ij  = A_ij * (dA_ij - delta_i)
//   dBias_ijh  = dlogit_ij
//   dQ_i       = scale * sum_j dlogit_ij k_j
//   dK_j       = scale * sum_i dlogit_ij q_i
//   dV_j       = sum_i A_ij dOAcc_i
//
// The backward is split into three entries so that each output is written by
// exactly one thread, with no atomics:
//
//   sml_attention_pair_bias_bwd_f32_stats    thread per (i,h): lse, oAcc, dOAcc, delta, dGate
//   sml_attention_pair_bias_bwd_f32_dq_dbias thread per (i,h): dQ, dBias
//   sml_attention_pair_bias_bwd_f32_dkdv     thread per (j,h): dK, dV
//
// Attention probabilities are recomputed from lse, so the forward output is not
// stored per pair.  Workspace (fp32): lse[N,H], oAcc[N,H,D], dOAcc[N,H,D], delta[N,H].
//
// Layout matches attention_pair_bias_fwd_f32:
//   q, k, v, gate, out, dOut, dQ, dK, dV, dGate : [N, H, D]   index (i*H + h)*D + d
//   bias, dBias                                 : [N, N, H]   index (i*N + j)*H + h
//
// Performance: the dq_dbias and dkdv entries each recompute N*H*N dot products.
// Register arrays hold D <= 64 floats per thread (see docs/af3_ops.md).

static const uint SML_PBB_MAX_D = 64u;

struct PairBiasAttentionBwdParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint D;
    float scale;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlPairBiasBwdPush;

float sml_pbb_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_pbb_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Pass 1: forward recomputation for one (i,h) row, plus the per-row gradients that
// depend only on that row.
[shader("compute")]
[numthreads(64, 1, 1)]
void sml_attention_pair_bias_bwd_f32_stats(uint3 dtid : SV_DispatchThreadID)
{
    PairBiasAttentionBwdParams* p = reinterpret<PairBiasAttentionBwdParams*>(smlPairBiasBwdPush.headerAddress);
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
    uint D = p.D;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;
    uint h = t % H;
    uint i = t / H;

    float qv[SML_PBB_MAX_D];
    float acc[SML_PBB_MAX_D];
    uint rowBase = (i * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_pbb_load(q, rowBase + d);
        acc[d] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint j = 0u; j < N; ++j)
    {
        uint keyBase = (j * H + h) * D;
        float dot = 0.0f;
        for (uint d = 0u; d < D; ++d)
            dot = fma(qv[d], sml_pbb_load(k, keyBase + d), dot);
        float logit = fma(dot, p.scale, sml_pbb_load(bias, (i * N + j) * H + h));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint d = 0u; d < D; ++d)
            acc[d] = fma(weight, sml_pbb_load(v, keyBase + d), acc[d] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    sml_pbb_store(lse, i * H + h, runningMax + log(runningSum));

    float deltaValue = 0.0f;
    for (uint d = 0u; d < D; ++d)
    {
        float g = slangml_sigmoid(sml_pbb_load(gate, rowBase + d));
        float o = acc[d] * invSum;
        float dO = sml_pbb_load(dOut, rowBase + d);
        float dOa = dO * g;
        sml_pbb_store(oAcc, rowBase + d, o);
        sml_pbb_store(dOAcc, rowBase + d, dOa);
        sml_pbb_store(dGate, rowBase + d, dO * o * g * (1.0f - g));
        deltaValue = fma(dOa, o, deltaValue);
    }
    sml_pbb_store(delta, i * H + h, deltaValue);
}

// Pass 2: dQ and dBias for one (i,h) row.
[shader("compute")]
[numthreads(64, 1, 1)]
void sml_attention_pair_bias_bwd_f32_dq_dbias(uint3 dtid : SV_DispatchThreadID)
{
    PairBiasAttentionBwdParams* p = reinterpret<PairBiasAttentionBwdParams*>(smlPairBiasBwdPush.headerAddress);
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dQ = slangml_tensor_desc(p.header, 7u);
    TensorDesc dBias = slangml_tensor_desc(p.header, 8u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;
    uint h = t % H;
    uint i = t / H;

    float qv[SML_PBB_MAX_D];
    float dO[SML_PBB_MAX_D];
    float dq[SML_PBB_MAX_D];
    uint rowBase = (i * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_pbb_load(q, rowBase + d);
        dO[d] = sml_pbb_load(dOAcc, rowBase + d);
        dq[d] = 0.0f;
    }
    float lseI = sml_pbb_load(lse, i * H + h);
    float deltaI = sml_pbb_load(delta, i * H + h);

    for (uint j = 0u; j < N; ++j)
    {
        uint keyBase = (j * H + h) * D;
        float dot = 0.0f;
        float dA = 0.0f;
        for (uint d = 0u; d < D; ++d)
        {
            dot = fma(qv[d], sml_pbb_load(k, keyBase + d), dot);
            dA = fma(dO[d], sml_pbb_load(v, keyBase + d), dA);
        }
        uint biasIndex = (i * N + j) * H + h;
        float logit = fma(dot, p.scale, sml_pbb_load(bias, biasIndex));
        float prob = exp(logit - lseI);
        float dlogit = prob * (dA - deltaI);
        sml_pbb_store(dBias, biasIndex, dlogit);
        float scaled = p.scale * dlogit;
        for (uint d = 0u; d < D; ++d)
            dq[d] = fma(scaled, sml_pbb_load(k, keyBase + d), dq[d]);
    }

    for (uint d = 0u; d < D; ++d)
        sml_pbb_store(dQ, rowBase + d, dq[d]);
}

// Pass 3: dK and dV for one (j,h) column.
[shader("compute")]
[numthreads(64, 1, 1)]
void sml_attention_pair_bias_bwd_f32_dkdv(uint3 dtid : SV_DispatchThreadID)
{
    PairBiasAttentionBwdParams* p = reinterpret<PairBiasAttentionBwdParams*>(smlPairBiasBwdPush.headerAddress);
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
    uint D = p.D;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;
    uint h = t % H;
    uint j = t / H;

    float kv[SML_PBB_MAX_D];
    float vv[SML_PBB_MAX_D];
    float dk[SML_PBB_MAX_D];
    float dv[SML_PBB_MAX_D];
    uint keyBase = (j * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        kv[d] = sml_pbb_load(k, keyBase + d);
        vv[d] = sml_pbb_load(v, keyBase + d);
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }

    for (uint i = 0u; i < N; ++i)
    {
        uint rowBase = (i * H + h) * D;
        float dot = 0.0f;
        float dA = 0.0f;
        for (uint d = 0u; d < D; ++d)
        {
            dot = fma(sml_pbb_load(q, rowBase + d), kv[d], dot);
            dA = fma(sml_pbb_load(dOAcc, rowBase + d), vv[d], dA);
        }
        uint biasIndex = (i * N + j) * H + h;
        float logit = fma(dot, p.scale, sml_pbb_load(bias, biasIndex));
        float prob = exp(logit - sml_pbb_load(lse, i * H + h));
        float dlogit = prob * (dA - sml_pbb_load(delta, i * H + h));
        float scaled = p.scale * dlogit;
        for (uint d = 0u; d < D; ++d)
        {
            float dOa = sml_pbb_load(dOAcc, rowBase + d);
            dv[d] = fma(prob, dOa, dv[d]);
            dk[d] = fma(scaled, sml_pbb_load(q, rowBase + d), dk[d]);
        }
    }

    for (uint d = 0u; d < D; ++d)
    {
        sml_pbb_store(dK, keyBase + d, dk[d]);
        sml_pbb_store(dV, keyBase + d, dv[d]);
    }
}

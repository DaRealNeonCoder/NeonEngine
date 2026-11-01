import slangml_common;

// Single-representation attention with pair bias and sigmoid output gate
// [SI Alg 24].  Used by the Pairformer single stream (16 heads) and by the
// diffusion token transformer (16 heads).  The q/k/v/gate projections, the
// LayerNorm(z) -> bias projection, and the output projection are host GEMMs.
//
// Layout (fp32, canonical):
//   q, k, v, gate, out : [N, H, D]   index (i*H + h)*D + d
//   bias               : [N, N, H]   index (i*N + j)*H + h   (already projected pair bias)
//
// out[i,h,:] = sigmoid(gate[i,h,:]) * sum_j softmax_j(q_i·k_j / sqrt(D) + bias[i,j,h]) v_j
//
// Thread mapping: one thread per (i, h).  Online softmax over j; the output
// accumulator lives in registers (D <= SML_ATTN_MAX_D).

static const uint SML_ATTN_MAX_D = 64u;

struct PairBiasAttentionParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint D;
    float scale;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlPairBiasAttentionPush;

float sml_attn_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_attention_pair_bias_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    PairBiasAttentionParams* p = reinterpret<PairBiasAttentionParams*>(smlPairBiasAttentionPush.headerAddress);
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc gate = slangml_tensor_desc(p.header, 4u);
    TensorDesc output = slangml_tensor_desc(p.header, 5u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;

    uint h = t % H;
    uint i = t / H;

    float qv[SML_ATTN_MAX_D];
    float acc[SML_ATTN_MAX_D];
    uint rowBase = (i * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_attn_load(q, rowBase + d);
        acc[d] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint j = 0u; j < N; ++j)
    {
        uint keyBase = (j * H + h) * D;
        float dot = 0.0f;
        for (uint d = 0u; d < D; ++d)
            dot = fma(qv[d], sml_attn_load(k, keyBase + d), dot);
        float logit = fma(dot, p.scale, sml_attn_load(bias, (i * N + j) * H + h));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint d = 0u; d < D; ++d)
            acc[d] = fma(weight, sml_attn_load(v, keyBase + d), acc[d] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    for (uint d = 0u; d < D; ++d)
    {
        float g = slangml_sigmoid(sml_attn_load(gate, rowBase + d));
        slangml_bda_store_f32(slangml_final_byte_address(output, int(rowBase + d), 4u), g * acc[d] * invSum);
    }
}
